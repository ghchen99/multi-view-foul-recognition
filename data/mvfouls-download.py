import os
import ssl
import certifi
from dotenv import load_dotenv
from SoccerNet.Downloader import SoccerNetDownloader as SNdl

ssl._create_default_https_context = ssl.create_default_context

load_dotenv()

def download_mvfouls_data(password):
    
    mySNdl = SNdl(LocalDirectory="data/")
    mySNdl.downloadDataTask(task="mvfouls", split=["train", "valid", "test", "challenge"], password=password)

if __name__ == "__main__":
    
    password = os.getenv("SOCCERNET_PASSWORD")
    if not password:
        raise ValueError("Password not found. Please set SOCCERNET_PASSWORD in your .env file.")
    
    download_mvfouls_data(password)
