import uuid


async def _upload(client, png_bytes):
    response = await client.post(
        "/api/v1/images", files={"file": ("scan.png", png_bytes, "image/png")}
    )
    return response.json()["id"]


async def test_process_image_runs_in_background_and_produces_a_result(client, png_bytes):
    image_id = await _upload(client, png_bytes)

    process = await client.post(
        f"/api/v1/images/{image_id}/process",
        json={"operation": "scale_image", "params": {"scale_factor": 2.0}},
    )
    assert process.status_code == 202
    job_id = process.json()["id"]
    assert process.json()["image_id"] == image_id

    # BackgroundTasks run as part of the ASGI response lifecycle (Starlette
    # awaits them before the send completes), so by the time the client call
    # above has returned, the job has already finished.
    finished = await client.get(f"/api/v1/jobs/{job_id}")
    assert finished.status_code == 200
    job = finished.json()
    assert job["status"] == "done"
    assert job["error"] is None
    assert job["result_image_id"] is not None

    result = await client.get(f"/api/v1/images/{job['result_image_id']}")
    assert result.status_code == 200
    result_body = result.json()
    assert result_body["parent_image_id"] == image_id
    # scale_image doubles a (height=3, width=4) source.
    assert result_body["width"] == 8
    assert result_body["height"] == 6

    downloaded = await client.get(f"/api/v1/images/{job['result_image_id']}/file")
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "image/png"


async def test_process_unknown_image_returns_404(client):
    response = await client.post(
        f"/api/v1/images/{uuid.uuid4()}/process",
        json={"operation": "scale_image", "params": {}},
    )
    assert response.status_code == 404


async def test_process_rejects_unknown_operation(client, png_bytes):
    image_id = await _upload(client, png_bytes)

    response = await client.post(
        f"/api/v1/images/{image_id}/process",
        json={"operation": "not_a_real_operation", "params": {}},
    )
    assert response.status_code == 422


async def test_get_unknown_job_returns_404(client):
    response = await client.get(f"/api/v1/jobs/{uuid.uuid4()}")
    assert response.status_code == 404
