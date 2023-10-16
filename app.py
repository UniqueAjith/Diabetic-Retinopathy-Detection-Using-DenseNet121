from flask import Flask, render_template, g,request, redirect, url_for, session, jsonify,send_file
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import base64
from PIL import Image
import io
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO
import numpy as np
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import img_to_array
from keras import models, layers
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE
from email import encoders
from datetime import datetime

app = Flask(__name__)
app.secret_key = '123564'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'Doctor'


mysql = MySQL(app)
@app.before_request
def load_user():
    if "id" in session:
        g.record = 1
        g.id = session['id']
        
    else:
        g.record = 0

@app.route('/', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE email = %s AND password = %s', (email, password,))
        account = cursor.fetchone()
        print(email,password)
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['firstname'] = account['firstname']
            session['lastname'] = account['lastname']
            session['email'] = account['email']
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute("""SELECT * FROM Patient WHERE doctorId = %s""", (session['id'],))
            data = cursor.fetchall()
            
            return render_template('doctor.html',name=session['firstname'],data=data)
        else:
            msg = 'Incorrect username/password!'
            
    return render_template('index.html')


# Logout
@app.route('/logout')
def logout():
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('firstname', None)

   return render_template('index.html')

@app.route('/Register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE email = %s', (email,))
        account = cursor.fetchone()

        if account:
            msg = 'Account already exists!'
        else:
           
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s,%s)', (firstname,lastname, password, email,))
            mysql.connection.commit()
            return render_template('register.html',)
    return render_template('register.html')

@app.route('/add_record',methods=['GET','POST'])
def add_record():
    if request.method == 'POST':
        name = request.form['name']
        age = int(request.form['age'])
        gender = request.form.get('gender')
        phone = request.form['phone']
        email = request.form['email']
        image = request.files['image'].read()
        
        status = predict(image)
        print(status)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM Patient WHERE email = %s', (email,))
        account = cursor.fetchone()
        if account:
            msg = 'Account Already exists'
        else:
            cursor.execute('INSERT INTO Patient VALUES (Null, %s, %s, %s,%s,%s,%s,%s,%s)', (name,int(age), gender, phone,email,status,image,int(g.id)))
            mysql.connection.commit()
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute("""SELECT PatientId,name,age,gender,phone,email,stage FROM Patient WHERE doctorId = %s""", (g.id,))
            data = cursor.fetchall()
            mysql.connection.commit()
            
            return render_template('doctor.html',data=data,message="Detected Successfully")
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("""SELECT * FROM Patient WHERE doctorId = %s""", (g.id,))
    data = cursor.fetchall()
    mysql.connection.commit()
    return render_template('doctor.html',name=session['firstname'],data=data)
 
@app.route('/update/<string:id>',methods = ['GET','POST'])
def update(id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("""SELECT * FROM Patient WHERE PatientId = %s""", (id,))
    data = cursor.fetchone()
    mysql.connection.commit()
    
    if request.method =="POST":
        name = request.form['name']
        print(name)
        age = int(request.form['age'])
        gender = request.form.get('gender')
        phone = request.form['phone']
        email = request.form['email']
        image = request.files['image'].read()
        
        status = predict(image)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("UPDATE Patient SET name = %s, age = %s,gender=%s,phone=%s,email=%s,stage=%s,image=%s WHERE PatientId = %s", (name,age,gender,phone,email,status,image,id,))
        mysql.connection.commit()
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("""SELECT * FROM Patient WHERE doctorId = %s""", (g.id,))
        data = cursor.fetchall()
        mysql.connection.commit()
        return render_template('doctor.html',name=session['firstname'],data=data)
    return render_template('update.html',name=session['firstname'],data = data)
@app.route('/delete/<string:id>')
def delete(id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(f"DELETE FROM Patient WHERE PatientId = {id}")
    cursor.execute("""SELECT * FROM Patient WHERE doctorId = %s""", (g.id,))
    data = cursor.fetchall()
    mysql.connection.commit()
    return render_template('doctor.html',name=session['firstname'],data=data)



def build_model():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(5, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def compile_model(model, learning_rate):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and compile the model
model = build_model()
model = compile_model(model, learning_rate=0.001)

# Load the saved weights into the model
model.load_weights('best_model.h5')
print('Model Loaded. Check this site http://127.0.0.1:5000')

# Define the predict function
def predict_output(image):
    # Open the image using PIL
    image = Image.open(BytesIO(image))
    # Resize and preprocess the image
    image = image.resize(size=(224,224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.
    # Run prediction on the image
    label = model.predict(image)[0]
    # Return the predicted label as JSON
    output =int(np.argmax(label))
    class_names = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    return class_names[output]

# Define a route for getting input image from user
@app.route('/predict', methods=['POST'])
def predict(img):
    # Get the input image from the request
    image = img
    # Run prediction on the image and return the result
    return predict_output(image)


now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

@app.route('/generate_certificate/<string:id>')
def generate_certificate(id):
    # Get the name and course from the request data
    print(id)
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(f"SELECT * FROM Patient WHERE PatientId = {id}")
    data = cursor.fetchall()
    mysql.connection.commit()
    res = data[0]
    PatientId = str(res['PatientId'])
    name = res['name']
    age = str(res['age'])
    gender = res['gender']
    phone = str(res['phone'])
    email = res['email']
    stage = res['stage']
    
    image = res['image']
    stage = str(stage)
    if stage == "Normal":
        Image = np.frombuffer(image, np.uint8)
        res = cv2.imdecode(Image, cv2.IMREAD_COLOR)
        res = cv2.resize(res, (412, 412))  # resize the image to 224 x 224
        img = cv2.imread("Negative-report.png")

    # Overlay the image onto the certificate at position (978, 1443)
        img[1078:1078+412, 1030:1030+412] = res
    #image.resize(size=(224,224))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, name, (281, 521), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, age, (1029, 521), font, 1, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img, gender, (281, 621), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, PatientId, (1029, 621), font, 1, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img, phone, (281, 721), font, 1, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img, email, (1029, 721), font, 1, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.putText(img, stage, (173, 1281), font, 2, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.putText(img, formatted_now, (285, 2172), font, 1, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.imwrite("report.png", img)
        
        return send_file("report.png", as_attachment=True)
    else:
        Image = np.frombuffer(image, np.uint8)
        res = cv2.imdecode(Image, cv2.IMREAD_COLOR)
        res = cv2.resize(res, (412, 412))  # resize the image to 224 x 224
        img = cv2.imread("Positive-report.png")

    # Overlay the image onto the certificate at position (978, 1443)
        img[1078:1078+412, 1030:1030+412] = res
    #image.resize(size=(224,224))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, name, (281, 521), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, age, (1029, 521), font, 1, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img, gender, (281, 621), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, PatientId, (1029, 621), font, 1, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img, phone, (281, 721), font, 1, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img, email, (1029, 721), font, 1, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.putText(img, stage, (173, 1281), font, 2, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.putText(img, formatted_now, (320, 2180), font, 1, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.imwrite("report.png", img)
        return send_file("report.png", as_attachment=True)
    
@app.route('/send_certificate/<string:id>')
def send_certificate(id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(f"SELECT * FROM Patient WHERE PatientId = {id}")
    data = cursor.fetchall()
    mysql.connection.commit()
    res = data[0]
    PatientId = str(res['PatientId'])
    name = res['name']
    age = str(res['age'])
    gender = res['gender']
    phone = str(res['phone'])
    email = res['email']
    stage = res['stage']
    
    image = res['image']
    stage = str(stage)
    if stage == "Normal":
        Image = np.frombuffer(image, np.uint8)
        res = cv2.imdecode(Image, cv2.IMREAD_COLOR)
        res = cv2.resize(res, (412, 412))  # resize the image to 224 x 224
        img = cv2.imread("Negative-report.png")

    # Overlay the image onto the certificate at position (978, 1443)
        img[1078:1078+412, 1030:1030+412] = res
    #image.resize(size=(224,224))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, name, (281, 521), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, age, (1029, 521), font, 1, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img, gender, (281, 621), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, PatientId, (1029, 621), font, 1, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img, phone, (281, 721), font, 1, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img, email, (1029, 721), font, 1, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.putText(img, stage, (173, 1281), font, 2, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.putText(img, formatted_now, (320, 2180), font, 1, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.imwrite("report.png", img)
        from_email = "gcesuaep@gmail.com"
        password = "terc awcj webf raiv"
        
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = COMMASPACE.join([email])
        msg['Subject'] = "Diabetic Retinopathy Result"

        body ="Hello {},\n\n Your ID number is: {}.\n\n  Diabetic Retinopathy Lab Report is generated.\n\n Please find the attachment.".format(name,id)
        msg.attach(MIMEText(body, 'plain'))

        filename = "certificate.png"
        attachment = open(filename, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

        msg.attach(part)

        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, email, msg.as_string())
        server.quit()
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("""SELECT * FROM Patient WHERE doctorId = %s""", (g.id,))
        data = cursor.fetchall()
        mysql.connection.commit()
        return render_template('doctor.html',name=session['firstname'],data=data)
        
    else:
        Image = np.frombuffer(image, np.uint8)
        res = cv2.imdecode(Image, cv2.IMREAD_COLOR)
        res = cv2.resize(res, (412, 412))  # resize the image to 224 x 224
        img = cv2.imread("Positive-report.png")

    # Overlay the image onto the certificate at position (978, 1443)
        img[1078:1078+412, 1030:1030+412] = res
    #image.resize(size=(224,224))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, name, (281, 521), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, age, (1029, 521), font, 1, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img, gender, (281, 621), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, PatientId, (1029, 621), font, 1, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img, phone, (281, 721), font, 1, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img, email, (1029, 721), font, 1, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.putText(img, stage, (173, 1281), font, 2, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.putText(img, formatted_now, (320, 2180), font, 1, (0, 0, 0), 2,cv2.LINE_AA)
        cv2.imwrite("report.png", img)
        from_email = "gcesuaep@gmail.com"
        password = "terc awcj webf raiv"
        
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = COMMASPACE.join([email])
        msg['Subject'] = "Diabetic Retinopathy Result"

        body = "Hello {},\n\n Your ID number is: {}.\n\n  Diabetic Retinopathy Lab Report is generated.\n\n Please find the attachment.".format(name,id)
        msg.attach(MIMEText(body, 'plain'))

        filename = "report.png"
        attachment = open(filename, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

        msg.attach(part)

        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, email, msg.as_string())
        server.quit()
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("""SELECT * FROM Patient WHERE doctorId = %s""", (g.id,))
        data = cursor.fetchall()
        mysql.connection.commit()
        return render_template('doctor.html',name=session['firstname'],data=data)
  
if __name__=='__main__':
    app.run(debug=True)
    
    
    
    
# CREATE TABLE Patient( PatientId INT PRIMARY KEY NOT NULL, name VARCHAR(50), age int, gender VARCHAR(255), phone VARCHAR(30), email VARCHAR(255), doctorId int );