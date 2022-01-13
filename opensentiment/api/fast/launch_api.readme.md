## Launch
cd ./opensentiment/api/fast
uvicorn serve_api:app --reload

## Try model
http://127.0.0.1:8000/serve_single?modelname=modelspecs_TODO&query_text=thisreviewisbad
http://127.0.0.1:8000/serve_single?modelname=modelspecs_TODO&query_text=%22this%20review%20is%20bad%22