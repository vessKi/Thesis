from ultralytics import YOLO, checks, hub
checks()

hub.login('68ef62e6abc72cfc50ad55cd52a4b14fe26b4d6256')

model = YOLO('https://hub.ultralytics.com/models/3jY57aCtAOeuUzyJ2M0F')
results = model.train()