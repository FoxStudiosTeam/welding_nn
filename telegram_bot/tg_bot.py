import telebot
from telebot import types
import time
from ultralytics import YOLO
import cv2
import numpy as np

from token import token
bot = telebot.TeleBot(token)

PATH = "../weights.pt"
model = YOLO(PATH)

TEXT_SIZE = 1.25
TEXT_THICKNESS = 2
TEXT_FACE = cv2.FONT_HERSHEY_SIMPLEX

LABELS = [
    "adj", 
    "int", 
    "geo", 
    "pro", 
    "non"
    ]

COLORS = [
    (47, 70, 238),
    (148, 155, 255),
    (27, 106, 255),
    (20, 181, 252),
    (61, 206, 207),
    ]

def relative_to_absolute(coords, IMAGE_SIZE):
    cx, cy, w, h = coords
    cx = float(cx)
    cy = float(cy)
    w = float(w)
    h = float(h)
    sx = (cx - w) * IMAGE_SIZE[0]
    sy = (cy - h) * IMAGE_SIZE[1]
    ex = (cx + w) * IMAGE_SIZE[0]
    ey = (cy + h) * IMAGE_SIZE[1]
    return int4((sx, sy, ex, ey))

def parse_label(path, IMAGE_SIZE):
    boxes = []
    for line in open(path, 'r'):
        i, cx, cy, w, h = line.split()
        boxes.append([
            int(i),
            relative_to_absolute((cx, cy, w, h), IMAGE_SIZE)
            ])
    return boxes

def parse_result(data, IMAGE_SIZE):
    boxes = []
    for box in data.boxes:
        boxes.append([
            int(box.cls),
            relative_to_absolute(box.xywhn.cpu().numpy().reshape(-1), IMAGE_SIZE)
            ])
    return boxes

def int4(coords):
    return int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

def draw_boxes(img, boxes):
    for box in boxes:
        i, coords = box
        sx, sy, ex, ey = coords
        cv2.rectangle(img, (sx, sy), (ex, ey), COLORS[i], 2)
    for box in boxes:
        i, coords = box
        sx, sy, ex, ey = coords
        text_size, _ = cv2.getTextSize(LABELS[i], TEXT_FACE, TEXT_SIZE, TEXT_THICKNESS)
        cv2.rectangle(img, (sx, sy), (sx + text_size[0], sy - text_size[1]), COLORS[i], -1)
        cv2.putText(img, LABELS[i], (sx, sy), TEXT_FACE, TEXT_SIZE, (0, 0, 0), TEXT_THICKNESS)
    return img


@bot.message_handler(commands=['start'])
def start_command(message):
    bot.send_message(message.chat.id, "Отправьте фото 📸")

@bot.message_handler(func=lambda m: True, content_types=['photo'])
def get_broadcast_picture(message):
    file_path = bot.get_file(message.photo[-1].file_id).file_path
    file = bot.download_file(file_path)
    img_array = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    result = model(img)
    predicted_image = result[0].orig_img
    IMAGE_SIZE = (predicted_image.shape[1], predicted_image.shape[0])
    predicted_boxes = parse_result(result[0], IMAGE_SIZE)
    predicted_image = draw_boxes(predicted_image, predicted_boxes)
    _, buffer = cv2.imencode('.jpg', predicted_image)
    bot.send_photo(message.chat.id, buffer.tobytes())
    if len(predicted_boxes) == 0:
        bot.send_message(message.chat.id, "Круто👍! Деффектов не найдено😍!")
    #bot.send_photo(message.chat.id, file)

bot.polling()
