from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    with open("received.jpg", "wb") as f:
        f.write(contents)
    return JSONResponse(content={"status": "success"})
