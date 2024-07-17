import cv2

def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        else:
            print(f"Camera index {index} works.")
            arr.append(index)
            # Open a live feed for this camera
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to grab frame for camera index {index}")
                    break
                cv2.imshow(f'Camera {index}', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        index += 1
    return arr

print("Listing available cameras:")
available_cameras = list_cameras()
print(f"Available cameras: {available_cameras}")
