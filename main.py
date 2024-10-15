import sqlite3      # veritabanı bağlantısı
import cv2          # görüntü işleme 
import insightface  # yüz tanıma
import os           # sistemsel çağrılar
import sys          # sistemsel çağrılar
import argparse     # argümanlar için
import numpy as np  # matematiksel işlemler için
import imghdr       # resim tanıma
import hashlib      # hash alma
import colorama     # renklendirme
from numba import njit # hızlandırma



# Windows konsollar için renklendirmenin başlatılması
colorama.init()


# print fonksiyonları
def p_info(msg:str) -> None:
    sys.stdout.write(f"{colorama.Fore.GREEN}[+]{colorama.Fore.RESET} {msg}\n")

def p_error(msg:str) -> None:
    sys.stderr.write(f"{colorama.Fore.RED}[+]{colorama.Fore.RESET} {msg}\n")



# Minimum benzerlik oranı tanıma için
MIN_SIMILARITY_RATE = 35
# Veritabanı dosya adı
DATABASE_NAME = "face_recognition.sqlite3"
# Veritabanı şeması
DATABASE_SCHEMA = """CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_name VARCHAR(200),
    face_pic BLOB,
    pic_hash VARCHAR(40),
    face_embeddings BLOB,
    face_age INT,
    face_gender BOOLEAN,
    add_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP  
);

CREATE INDEX idx_id_faces ON faces (id);
CREATE INDEX idx_face_name_faces ON faces (face_name);
CREATE INDEX idx_face_pic_faces ON faces (face_pic);
CREATE INDEX idx_pic_hash_faces ON faces (pic_hash);
CREATE INDEX idx_face_embeddings_faces ON faces (face_embeddings);
CREATE INDEX idx_face_age_faces ON faces (face_age);
CREATE INDEX idx_face_gender_faces ON faces (face_gender);



"""

p_info(f"Connecting database: {DATABASE_NAME}")

# Veritabanı bağlantısının başlatılması
db = sqlite3.connect(database=DATABASE_NAME,check_same_thread=False)
cursor = db.cursor()

# Yüz tanıma için kullanılan insightface kütüphanesinin başlatılması ve CUDA nın ayarlanmas
insightfaceApp = insightface.app.FaceAnalysis(providers=["CUDAExecutionProvider"],name="antelopev2",)
insightfaceApp.prepare(ctx_id=0,det_size=(640,640))


try:
    cursor.executescript(DATABASE_SCHEMA)
    db.commit()
    p_info(f"Database schema executed.")
except sqlite3.OperationalError as err:
    pass


def landmarks_rectangle(cv2_image:np.ndarray, data_list:list, face_name:str) -> np.ndarray:
    left, top, right, bottom = map(int, data_list)
    cv2.rectangle(cv2_image, (left, top), (right, bottom), (0, 255, 0), 3)
    
    text = f"{face_name}"
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1
    font_thickness = 2
    color = (0, 0, 255)
    
    cv2.putText(frame,text,(left,top-10),font,font_scale,color,font_thickness)
    
    return cv2_image


def landmarks_rectangle_2d(cv2_image:np.ndarray, data_list:list) -> np.ndarray:
    for landmark_point in data_list:
        x,y = map(int, landmark_point)
        cv2.circle(cv2_image, (x,y),1, (0,255,0), -1)
        
    return cv2_image

@njit(nopython=True)
def compute_cosine_sim(source:np.ndarray, target:np.ndarray) -> float:
        dot_product_size = np.dot(source, target)
        norm_sound1 = np.linalg.norm(source)
        norm_sound2 = np.linalg.norm(target)

        # kosinus benzerliğini hesaplama 
        GetSimilarity = dot_product_size / (norm_sound1 * norm_sound2)
        return GetSimilarity

@njit(nopython=True)
def buffer2numpy_float32(buffer_data:bytes) -> np.ndarray:
    """
    Args:
        buffer_data (bytes): blob numpy array from postgresql database 

    Returns:
        numpy.ndarray: usable numpy array for python3 
    """
    
    _data = np.frombuffer(buffer_data,dtype=np.float32)
    return _data





argParser = argparse.ArgumentParser()
argParser.add_argument("--add-face",required=False)
argParser.add_argument("--name",required=False)
argParser.add_argument("--cam",required=False)
argParser.add_argument("--res",type=str,required=False,help="Resulation",default="1200,720")
argParser.add_argument("--delete-face",type=str,required=False)
argParser.add_argument("--dump-db",type=bool,required=False)

argsIs = vars(argParser.parse_args())

if argsIs["add_face"] != None and argsIs["name"] != None:
    faceName = argsIs["name"]
    facePicPath = argsIs["add_face"]
    facePicData = None
    opencvImage = None
    imageHash = None
    faceData = None
    faceAge = None
    faceGender = None
    faceEmbeddings = None
    
    if not os.path.isfile(facePicPath):
        p_error(f"Image file not found: {facePicPath}")
        sys.exit(1)
        
    
    with open(facePicPath, "rb") as imageFile:
        if imghdr.what(None,imageFile.read(512)).lower() not in ["jpg", "png", "jpeg", "wepm"]:
            p_error(f"Invalid image file: {facePicPath}")
            sys.exit(1)
            
        imageFile.seek(0)    
        facePicData = imageFile.read()
        imageHash = hashlib.sha1(facePicData).hexdigest()
    
    opencvImage = cv2.imdecode(np.frombuffer(facePicData,np.uint8), cv2.IMREAD_COLOR)

    if not opencvImage.any():
        p_error(f"Failed to read image: {facePicPath}")
        sys.exit(1)
    

    faceData = insightfaceApp.get(opencvImage)
    
    if len(faceData) != 1:
        p_error(f"Allowed face count is 1 but your image have {str(len(faceData))} faces.")
        sys.exit(1)
    
    faceData = faceData[0]
    
    faceGender = faceData["gender"]
    faceAge = faceData["age"]
    faceEmbeddings = faceData["embedding"]
    
    STATIC_SQL_COMMAND = """INSERT INTO faces (face_name, face_pic, pic_hash, face_embeddings, face_age, face_gender) VALUES (?,?,?,?,?,?)"""
    STATIC_DATA_TUPLE = (faceName, facePicData, imageHash, faceEmbeddings, faceAge, bool(faceGender))

    cursor.execute(STATIC_SQL_COMMAND, STATIC_DATA_TUPLE)
    db.commit()
    
    p_info(f"Your face successfuly added: {faceName}")
    sys.exit(0)
    
    
elif argsIs["cam"]:    
    camID = argsIs["cam"]
    resulation = argsIs["res"]

    if resulation == None:
        resulation = "1280,720"

    if "," not in resulation or len(str(resulation).split(",")) != 2 or not str(resulation).split(",")[0].isnumeric() or not str(resulation).split(",")[1].isnumeric() :
        p_error(f"Invalid Resulation...")
        sys.exit(1)
        
    resulation = list(str(resulation).split(","))
    resulation[0] = int(resulation[0])
    resulation[1] = int(resulation[1])
    resulation = tuple(resulation)

    
    if str(camID).isnumeric():
        camID = int(camID)
    else:
        camID = str(camID)

    camera = cv2.VideoCapture(camID)
    
    if not camera.isOpened():
        p_error(f"Failed to open camera: {camID}")
        sys.exit(1)

    badCounter = 0
    frameCounter = 0
        
    
    STATIC_SQL_COMMAND = "SELECT * FROM faces"
    cursor.execute(STATIC_SQL_COMMAND)
    
    p_info(f"Loading face database to memory ....")
    allFaceData = cursor.fetchall()
    p_info(f"Load success, total face data: {str(len(allFaceData))}")
        
        
    while True:
        
        ret_code, frame = camera.read()
        
        if ( badCounter % camera.get(cv2.CAP_PROP_FPS) ) == 10:
            p_error(f"No vdieo for 10 seconds, exiting ...")
            break
        
        if not ret_code:
            badCounter += 1
            continue
        
        frame = cv2.resize(frame,resulation)
        detectData = insightfaceApp.get(frame)
            
        if len(detectData) > 0:
            #app.draw_on(frame,detectData)
            for face in detectData:
                for singleFace in allFaceData:
                    dbEmbeddings = buffer2numpy_float32(singleFace[4])
                    currentEmbeddings = face["embedding"]

                    similarityRate = compute_cosine_sim(dbEmbeddings, currentEmbeddings)
                    similarityRate = int(similarityRate*100)
                    
                    if similarityRate >= MIN_SIMILARITY_RATE:
                        faceName = f"%{similarityRate} | {singleFace[1]}"
                    else:
                        faceName = ""    
                    
                    frame = landmarks_rectangle(frame, data_list=face["bbox"],face_name=faceName)                    
                    #frame = landmarks_rectangle_2d(cv2_image=frame,data_list=face["landmark_2d_106"])

                    
        cv2.imshow(f"Camera: {camID}", frame)
        
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()

elif argsIs["delete_face"]:
    deleteTarget = argsIs["delete_face"]
    p_info(f"Deleting face: {str(deleteTarget)}")
    
    
    STATIC_SQL_COMMAND = """SELECT * FROM faces WHERE face_name=?"""
    STATIC_DATA_TUPLE = (str(deleteTarget), )
    
    cursor.execute(STATIC_SQL_COMMAND, STATIC_DATA_TUPLE)
    totalDeleteFaceCount = len(cursor.fetchall())
    
    
    if totalDeleteFaceCount == 0:
        p_error(f"No match found for face name: {str(deleteTarget)}")
        sys.exit(0)
        
    p_error(f"Deleting {totalDeleteFaceCount} faces")
    
    STATIC_SQL_COMMAND = """DELETE FROM faces WHERE face_name = ?"""
    STATIC_DATA_TUPLE = (str(deleteTarget), )
    
    cursor.execute(STATIC_SQL_COMMAND, STATIC_DATA_TUPLE)
    db.commit()
    
    p_error(f"Proccess success")
    
    
elif argsIs["dump_db"]:
    STATIC_SQL_COMMAND = "SELECT id, face_name, face_age, face_gender,pic_hash, add_date FROM faces"
    
    p_info("Executing query...")
    
    cursor.execute(STATIC_SQL_COMMAND)
    results = cursor.fetchall()
    
    p_info(f"Total results: {str(len(results))}")
    
    print()
    
    for single in results:
        id_is = str(single[0])
        face_name = str(single[1])
        face_age = str(single[2])
        face_gender = str(single[3])
        pic_hash = str(single[4])
        add_date = str(single[5])
        
        print("-"*50)
        print(f"ID: {id_is}\nFACE NAME: {face_name}\nFACE AGE: {face_age}\nFACE GENDER: {face_gender}\nPICTURE HASH: {pic_hash}\nADD DATE: {add_date}")
    print("-"*50)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    














    

    
    
    
    
    
    
    
    
    






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
            
    
    






























































