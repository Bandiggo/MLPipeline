import pandas as pd
from fastapi import FastAPI
import basic_pipeline

app = FastAPI()


@app.get("/")
async def root():
    basic_pipeline.getNoOfCalls(pd.read_csv('/Users/ezgi-lab/MLPipeline/data/atasehir.csv'))
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
