import uvicorn
from os import getenv


if __name__ == "__main__":
    port = int(getenv("PORT",3000))
    uvicorn.run("api:app",host="localhost",port=port,reload=True)