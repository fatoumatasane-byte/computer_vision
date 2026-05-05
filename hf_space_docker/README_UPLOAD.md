Upload these files to your Hugging Face Docker Space root:

1) app.py
2) requirements.txt
3) Dockerfile
4) fatima_model.pth
5) fatima_model.keras
6) fatima_meta.json (optional but recommended)

Notes:
- If your model filenames differ, update PT_MODEL_PATH / TF_MODEL_PATH in app.py.
- If you do not have fatima_meta.json, the app uses default class names and ImageNet normalization.
- For Docker Space, keep app.py launch as:
  demo.launch(server_name="0.0.0.0", server_port=7860)
