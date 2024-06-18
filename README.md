# Hear4U ML Server
This repository contains the server-side code for deploy and serving ML model.

## Deployed Link
http://34.101.212.39:8000/

## Techstack
Some several technologies or libraries that we used:
<ol>
  <li>FastAPI</li>
  <li>Tensorflow</li>
  <li>Socket.IO</li>
  <li>Pydub</li>
  <li>Compute Engine</li>
</ol>

## Get Started
Clone `repository` with command:
```
git clone https://github.com/HEAR4U-bangkit/cc-hear4u-server.git
```

Install dependencies:
```
pip install -r requirements.txt
```

Start the server with command:
```
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then can be accessed in:
```
<IP ADDRESS>:8000
```

