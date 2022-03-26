import argparse
import cv2
import pickle
import tensorflow as tf
import numpy as np

from yolo import YOLO

# From https://stackoverflow.com/questions/11540854/file-as-command-line-argument-for-argparse-error-message-if-argument-is-not-va
import os.path
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", choices=["normal", "tiny", "prn", "v4-tiny"],
                help='Network Type')
ap.add_argument('-d', '--device', type=int, default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
ap.add_argument('-nh', '--hands', default=-1, help='Total number of hands to be detected per frame (-1 for all)')
ap.add_argument('-m', '--model', type=lambda x: is_valid_file(ap, x),
    required=True, help='Keras model to use on the output of YOLO')
ap.add_argument('-md', '--model_dict', type=lambda x: is_valid_file(ap, x),
    required=True, help='Dictionnary to map the numerical output to an alphabetic one')

args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("loading the model ..")
model = tf.keras.models.load_model(args.model.name)
with open(args.model_dict.name, 'rb') as f:
    mapping_d = pickle.load(f)
input_shape = model.layers[0].input_shape[1:]
is_mnist = (input_shape[0] == 28)
is_color = (input_shape[-1] == 3)
print('Input shape = ', str(input_shape))
print('is_mnist = ', str(is_mnist))
print('is_color = ', str(is_color))
print('groups = ', str(mapping_d.values()))
if len(model.layers) > 5 and not is_mnist: # Resnet
    preprocessing_f = tf.keras.applications.resnet50.preprocess_input
elif not is_mnist: # First bad model
    preprocessing_f = lambda x : x / 255.
else: # MNIST model
    preprocessing_f = lambda x : np.expand_dims(np.average(x, axis=-1, weights=(0.299, 0.587, 0.114)) / 255., axis=-1)

print("starting webcam...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(args.device)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

i = 0
while rval:
    width, height, inference_time, results = yolo.inference(frame)

    # display fps
    cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,255), 2)

    # sort by confidence
    results.sort(key=lambda x: x[2])

    # how many hands should be shown
    hand_count = len(results)
    if args.hands != -1:
        hand_count = int(args.hands)

    # display hands
    for detection in results[:hand_count]:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        # BEGIN HACK

        # Select frame but as a good square
        # The hand takes the whole space
        d = int((w - h) / 2)
        if d > 0:
            ymin,ymax,xmin,xmax = y-d, y+h+d, x, x+w
        else:
            ymin,ymax,xmin,xmax = y, y+h, x+d, x+w-d


        # Add 50% of extra on borders
        if not is_mnist:
            dy = ymax - ymin
            dx = xmax - xmin
            ymin -= int(dy/4)
            ymax += int(dy/4)
            xmin -= int(dx/4)
            xmax += int(dx/4)

        # Crop the image
        ymin = max(ymin, 0)
        xmin = max(xmin, 0)
        ymax = min(ymax, frame.shape[0])
        xmax = min(xmax, frame.shape[1])
        frame_cropped = frame[ymin:ymax, xmin:xmax]

        #cv2.imwrite('tmp/image_B{}.png'.format(i), frame_cropped)
        #i += 1

        # Ensure frames are squares
        frame_cropped = preprocessing_f(frame_cropped[:min(frame_cropped.shape[:2]), :min(frame_cropped.shape[:2])])

        # Drop the frame if it's too small
        if frame_cropped.size < (input_shape[0] * input_shape[1] * input_shape[2])/2:
            continue

        # Make it a batch
        frame_cropped_b = np.expand_dims(frame_cropped, axis=0)

        # Resize it 
        frame_cropped_b = tf.image.resize(frame_cropped_b, input_shape[:2])

        # Flip it
        frame_cropped_b = tf.image.flip_left_right(frame_cropped_b)

        # Run through the model
        sign_scores = model.predict(frame_cropped_b)[0]
        letter_idx = np.argmax(sign_scores)

        # Overwrite name and confidence
        name = mapping_d[letter_idx]
        confidence = sign_scores[letter_idx]

        # END HACK
        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    cv2.imshow("preview", frame)

    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
