import uuid


async def test_upload_list_get_and_download_image(client, png_bytes):
    upload = await client.post(
        "/api/v1/images",
        files={"file": ("scan.png", png_bytes, "image/png")},
    )
    assert upload.status_code == 201
    body = upload.json()
    assert body["filename"] == "scan.png"
    assert body["content_type"] == "image/png"
    assert body["width"] == 4
    assert body["height"] == 3
    assert body["parent_image_id"] is None
    image_id = body["id"]

    listing = await client.get("/api/v1/images")
    assert listing.status_code == 200
    assert [img["id"] for img in listing.json()] == [image_id]

    fetched = await client.get(f"/api/v1/images/{image_id}")
    assert fetched.status_code == 200
    assert fetched.json() == body

    downloaded = await client.get(f"/api/v1/images/{image_id}/file")
    assert downloaded.status_code == 200
    assert downloaded.content == png_bytes
    assert downloaded.headers["content-type"] == "image/png"


async def test_upload_rejects_undecodable_bytes(client):
    response = await client.post(
        "/api/v1/images",
        files={"file": ("not-an-image.png", b"definitely not a png", "image/png")},
    )
    assert response.status_code == 422


async def test_get_unknown_image_returns_404(client):
    response = await client.get(f"/api/v1/images/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_get_unknown_image_file_returns_404(client):
    response = await client.get(f"/api/v1/images/{uuid.uuid4()}/file")
    assert response.status_code == 404


async def test_delete_unknown_image_returns_404(client):
    response = await client.delete(f"/api/v1/images/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_delete_image_removes_it(client, png_bytes):
    upload = await client.post(
        "/api/v1/images", files={"file": ("scan.png", png_bytes, "image/png")}
    )
    image_id = upload.json()["id"]

    delete = await client.delete(f"/api/v1/images/{image_id}")
    assert delete.status_code == 204

    assert (await client.get(f"/api/v1/images/{image_id}")).status_code == 404
    assert (await client.get("/api/v1/images")).json() == []
