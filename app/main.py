from typing import List
from os import path, getcwd
import sys

import cv2 as cv
import numpy as np

def print_help():
  print("Usage: main.py <input image>")

def detect_face_with_jaws(image: np.ndarray) -> np.ndarray:
  faces_with_jaws = []
  RADIUS = 30 # "jaw" point detection must be in RADIUS proximity of skin edge
  mat_3_3 = np.ones((3,3), np.uint8)
  mat_5_5 = np.ones((5,5), np.uint8)

  image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
  skin_threshold = cv.inRange(image_hsv, (0, 0, 75*255//100), (30, 255, 255))
  white_threshold = cv.inRange(image_hsv, (0,0, 75*255/100), (0,255,255))
  skin_threshold = cv.bitwise_xor(skin_threshold, white_threshold) # remove white from skin
  skin_threshold = cv.morphologyEx(skin_threshold, cv.MORPH_OPEN, mat_3_3) # erode then dilate
  skin_masked = cv.bitwise_and(image, image, mask=skin_threshold) # mask image using skin color
  skin_edges = cv.Canny(skin_masked, 150, 210, apertureSize=3) #edges of 
  skin_edges = cv.morphologyEx(skin_edges, cv.MORPH_CLOSE, mat_5_5) # dilate then erode
  skin_contours, _ = cv.findContours(skin_edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # get contours of skin edges

  for sc in skin_contours:
    x, y, w, h = cv.boundingRect(sc)
    if w < 50 or h < 50 or not (2/5 < w/h < 7/3): # arbitrary numbers, need more experimentations.
      continue
    mask = np.zeros(skin_threshold.shape, dtype=np.uint8) # for masking the contour region
    mask = cv.drawContours(mask, [sc], 0, (255), 1)
    mask = cv.dilate(mask, mat_3_3)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, mat_3_3)
    edges_skin = cv.Canny(cv.bitwise_and(skin_threshold, skin_threshold, mask=mask), 150, 210, apertureSize=5) # get edges of current region

    lines = cv.HoughLines(edges_skin, 1, np.pi*1/180, 50, None, 0, 0) # get lines using hough transform
    if lines is not None and lines.shape[0] > 1:
      f_lines = []
      lines = lines.reshape((-1,2))
      # segment lines based on quadrant
      lines_q1 = lines[(lines[:,1] < np.deg2rad(90)).nonzero()]
      lines_q2 = lines[(lines[:,1] >= np.deg2rad(90)).nonzero()]
      for l in lines_q1:
        for ll in lines_q2:
          rho_i, theta_i = l.tolist()
          rho_j, theta_j = ll.tolist()
          # check if lines intersect at a certain angle
          if np.deg2rad(50) < np.abs(theta_i - theta_j) < np.deg2rad(100):
            # if yes, solve for intersection point
            A = np.array([
              [np.cos(theta_i), np.sin(theta_i)],
              [np.cos(theta_j), np.sin(theta_j)]
            ])
            b = np.array([[rho_i], [rho_j]])
            x0, y0 = np.linalg.solve(A, b)
            # the point could be considered a "jaw" point if it fulfills:
            # sanity check: it should be inside the image frame
            if 0 <= x0 <= image.shape[1] and 0 <= y0 <= image.shape[0]:
              # it should be near a skin edge
              nz = (skin_edges[int(x0)-RADIUS:int(x0)+RADIUS, int(y0)-RADIUS:int(y0)+RADIUS]).any()
              if nz:
                f_lines.append([int(np.round(x0)), int(np.round(y0))])
      if len(f_lines) > 0:
        x_jaw, y_jaw = np.array(f_lines).mean(axis=0)
        faces_with_jaws.append([sc, [x_jaw, y_jaw]])
  return faces_with_jaws

def detect_eyes(image: np.ndarray, contour: List, jaw: List[int]) -> np.ndarray:
  x0, y0, w, h = cv.boundingRect(contour)
  jaw_corrected = [jaw[0]-x0, jaw[1]-y0]
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 3
  Z = np.float32(image[y0:y0+h,x0:x0+w].reshape((-1, 3)))
  _, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

  kern_3_3 = np.ones((3,3))

  center = np.uint8(center)
  percentage = [np.count_nonzero(label == i)/len(label) for i in range(3)]
  min_idx = np.argmin(percentage)
  res = center[label.flatten()]
  res[~(label == min_idx).flatten()] = (0,0,0)
  res[(label == min_idx).flatten()] = (255,255,255)
  res = res.reshape((h,w,3))
  gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
  _, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
  binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kern_3_3)
  binary = cv.dilate(binary, kern_3_3)
  contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  potential_eyes = []
  for c in contours:
    # loop through all contours, find potential eye contour
    _, _, w, h = cv.boundingRect(c)
    if (w < 10) or h < 10 or not\
      ((1 < w/h < 2.5) or\
      (1.5 < h/w < 2)):
      continue
    potential_eyes.append(c)

  eyes = []
  offset = [x0, y0]

  for i,e1 in enumerate(potential_eyes[:-1]):
    x1, y1, w1, h1 = cv.boundingRect(e1)
    center1 = np.array([x1+w1/2, y1+h1/2])
    jaw_to_e1 = np.linalg.norm(jaw_corrected - center1)
    for e2 in potential_eyes[i+1:]:
      sim = cv.matchShapes(e1, e2, cv.CONTOURS_MATCH_I1, 0.0)
      if sim > 0.325:
        continue
      x2, y2, w2, h2 = cv.boundingRect(e2)

      eye1 = res[y1:y1+h1, x1:x1+w1]
      eye2 = res[y2:y2+h2, x2:x2+w2]

      eye1 = cv.cvtColor(eye1, cv.COLOR_BGR2HSV)
      eye2 = cv.cvtColor(eye2, cv.COLOR_BGR2HSV)

      eye1 = eye1.reshape((-1, 3))
      eye2 = eye2.reshape((-1, 3))
      av1 = np.average(eye1, axis=0)
      av2 = np.average(eye2, axis=0)
      diff = np.abs(av1 - av2)
      if (diff > (45, 65, 50)).all():
        continue
      center2 = np.array([x2+w2/2, y2+h2/2])
      jaw_to_e2 = np.linalg.norm(jaw_corrected - center2)
      if np.abs(jaw_to_e1 - jaw_to_e2) > 40:
        # distance "between eyes to jaw" too big
        continue
      eyes.append([offset+center1, offset+center2])
      break
  return eyes

def detect_landmarks(image: np.ndarray):
  faces_with_jaws = detect_face_with_jaws(image)
  zipped = list(zip(faces_with_jaws, [detect_eyes(image, f[0], f[1]) for f in faces_with_jaws]))
  return list(zipped)

def main(argv: List[str]):
  if len(argv) < 2:
    print_help()
    sys.exit(1)
  
  image = cv.imread(path.join(getcwd(), argv[1]))
  face_landmarks = detect_landmarks(image)
  for f in face_landmarks:
    x, y, w, h = cv.boundingRect(f[0][0])
    cv.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
    for eyes in f[1]:
      for eye in eyes:
        cv.circle(image, (int(eye[0]), int(eye[1])), 3, (0,0,255), -1) # draw eyes
    cv.circle(image, (int(f[0][1][0]), int(f[0][1][1])), 3, (0,0,255), -1) # draw jaw
  
  cv.imshow('Result', image)
  cv.waitKey(0)
  cv.destroyAllWindows()

if __name__ == "__main__":
  main(sys.argv)
