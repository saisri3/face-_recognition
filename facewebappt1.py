import os 
import sys
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import face_recognition
import cv2
import numpy as np

UPLOAD_FOLDER = '/home/saisri/projectcopy_5/project/upload'
ALLOWED_EXTENSIONS = set(['txt','mp4'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            NEWPATH=videotest(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('video.html')
        
    return render_template('indext1.html')


    



def videotest(filename):
    video_capture = cv2.VideoCapture(filename)
    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    sai = face_recognition.load_image_file("2.jpg")
    sfencoding = face_recognition.face_encodings(sai)[0]

    dhoni = face_recognition.load_image_file("1.jpg")
    dfencoding = face_recognition.face_encodings(dhoni)[0]
    
    virat = face_recognition.load_image_file("3.jpg")
    vfencoding = face_recognition.face_encodings(virat)[0]
    
    modi = face_recognition.load_image_file("4.jpg")
    mfencoding = face_recognition.face_encodings(modi)[0]
    
    prateek = face_recognition.load_image_file("prateek.jpg")
    pfencoding = face_recognition.face_encodings(prateek)[0]
    
    harmesh = face_recognition.load_image_file("harmesh.jpg")
    hfencoding = face_recognition.face_encodings(harmesh)[0]
    
    vishnu = face_recognition.load_image_file("vishnu.jpg")
    nvencoding = face_recognition.face_encodings(vishnu)[0]

    robert = face_recognition.load_image_file("robertdowney jr.jpg")
    rbjencoding = face_recognition.face_encodings(robert)[0]
        
    harry = face_recognition.load_image_file("daniel radcliffe.jpg")
    harryencoding = face_recognition.face_encodings(harry)[0]
    
    hermoine = face_recognition.load_image_file("emma watson.jpg")
    herencoding = face_recognition.face_encodings(hermoine)[0]

    rupert = face_recognition.load_image_file("ront1.jpg")
    ronencoding = face_recognition.face_encodings(rupert)[0]
        
    
    
    

   
    known_face_encodings = [
        sfencoding,
        dfencoding,
        vfencoding,
        mfencoding,
        pfencoding,
        hfencoding,
        nvencoding,
        rbjencoding,
        harryencoding,
        herencoding,
        ronencoding
    ]
    known_face_names = [
        
        "sai",
        "DHONI",
        "virat",
        "modi",
        "prateek",
        "harmesh",
        "vishnu",
        "robert downey jr.",
        "daniel radcliffe",
        "emma watson",
        "rupert grint"
    ]

    width  = int(video_capture.get(3)) # float
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    PATH = '/home/saisri/projectcopy_5/project/demo.webm'
    out = cv2.VideoWriter(PATH,fourcc, fps, (width,height))
    for i in range(1,length-1):
        
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 10), (right, bottom + 10 ), (10, 10, 10), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 2, bottom), font, 0.4, (255, 255, 255), 1)

        print()
        sys.stdout.write(f"writing...{int((i/length)*100)+1}%")
        sys.stdout.flush()
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    return PATH

    
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5002, debug=False)
