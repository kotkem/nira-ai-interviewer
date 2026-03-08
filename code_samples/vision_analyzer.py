import cv2
import math
import mediapipe as mp
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

class VisionAnalyzer(QThread):
    frame_ready = pyqtSignal(QImage)
    raw_frame_ready = pyqtSignal(QImage) 

    def __init__(self):
        super().__init__()
        self.running = True

        # Variables para el parpadeo
        self.blink_counter = 0
        self.is_blinking = False
        self.current_attention = "Alta"
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True
        )
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

        # Estilos Nira (Nueva Paleta Dorado/Morado)
        self.nira_lime = (255, 215, 0)      # Dorado Principal
        self.nira_cyan = (200, 180, 0)      # Dorado oscuro para contornos
        self.nira_purple = (160, 32, 240)   # Morado brillante
        self.nira_magenta = (138, 43, 226)  # Morado oscuro
        self.nira_red = (255, 50, 50)       # Rojo para alertas

        self.mesh_style = self.mp_drawing.DrawingSpec(color=self.nira_lime, thickness=1, circle_radius=0)
        self.contour_style = self.mp_drawing.DrawingSpec(color=self.nira_cyan, thickness=1, circle_radius=0)
        self.hand_joint_style = self.mp_drawing.DrawingSpec(color=self.nira_purple, thickness=2, circle_radius=2)
        self.hand_bone_style = self.mp_drawing.DrawingSpec(color=self.nira_lime, thickness=2)
        self.pose_joint_style = self.mp_drawing.DrawingSpec(color=self.nira_magenta, thickness=2, circle_radius=3)
        self.pose_bone_style = self.mp_drawing.DrawingSpec(color=self.nira_cyan, thickness=2)
        
    def _calculate_distance(self, p1, p2):
        """Calcula la distancia Euclidiana entre dos puntos 2D."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def _analyze_blinks(self, face_landmarks):
        """Calcula el Eye Aspect Ratio (EAR) para detectar parpadeos."""
        # Puntos del ojo derecho en MediaPipe
        p_top = face_landmarks.landmark[159]
        p_bottom = face_landmarks.landmark[145]
        p_left = face_landmarks.landmark[33]
        p_right = face_landmarks.landmark[133]

        # Distancias
        dist_vertical = self._calculate_distance(p_top, p_bottom)
        dist_horizontal = self._calculate_distance(p_left, p_right)

        if dist_horizontal == 0:
            return

        # Ratio del ojo
        ear = dist_vertical / dist_horizontal

        # Lógica para contar el parpadeo (si el ratio cae por debajo de 0.22, el ojo está cerrado)
        if ear < 0.22:
            if not self.is_blinking:
                self.is_blinking = True
                self.blink_counter += 1
        else:
            self.is_blinking = False

    def _analyze_attention(self, face_landmarks):
        """Analiza la orientación de la cabeza para determinar la atención."""
        # Puntos clave extraídos de MediaPipe 
        nose_tip = face_landmarks.landmark[1]
        left_eye_outer = face_landmarks.landmark[33]
        right_eye_outer = face_landmarks.landmark[263]

        # Calcular distancias relativas
        dist_left = self._calculate_distance(nose_tip, left_eye_outer)
        dist_right = self._calculate_distance(nose_tip, right_eye_outer)

        # Evitar división por cero
        if dist_right == 0:
            return "ERROR", self.nira_red

        ratio = dist_left / dist_right

        # Si el ratio está cerca de 1.0, la cara está centrada
        if 0.75 < ratio < 1.35:
            return "ATENCION: ALTA (Mirando al frente)", self.nira_lime
        else:
            return "ATENCION: BAJA (Mirada desviada)", self.nira_red

    def run(self):
        cap = cv2.VideoCapture(0)

        while self.running and cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Invertir la imagen como un espejo
            image = cv2.flip(image, 1)
            
            # --- APLICAR DESENFOQUE VIRTUAL---
            image_rgb_tmp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_seg = self.segmentation.process(image_rgb_tmp)
            condition = np.stack((results_seg.segmentation_mask,) * 3, axis=-1) > 0.5
            
            # Difuminado profundo del fondo
            bg_image = cv2.GaussianBlur(image, (75, 75), 0)
            
            image = np.where(condition, image, bg_image)

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image_rgb)
            image.flags.writeable = True
            
            # --- ENVÍO A INTERFAZ ---
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w
            raw_qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            self.raw_frame_ready.emit(raw_qt_image)
            image_rgb_m = image_rgb.copy()

            # Variable para guardar el texto del HUD
            hud_text = "Buscando candidato..."
            text_color = self.nira_cyan

            # --- PROCESAMIENTO DE ROSTRO Y ANÁLISIS ---
            if results.face_landmarks:
                # Dibujar Mallas
                self.mp_drawing.draw_landmarks(image_rgb_m, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, None, self.mesh_style)
                self.mp_drawing.draw_landmarks(image_rgb_m, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS, None, self.contour_style)
                
                # Extraer Métricas
                hud_text, text_color = self._analyze_attention(results.face_landmarks)
                self._analyze_blinks(results.face_landmarks) # Llamamos al detector de parpadeo

            # --- DIBUJAR RESTO DEL CUERPO ---
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(image_rgb_m, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, self.pose_joint_style, self.pose_bone_style)
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(image_rgb_m, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.hand_joint_style, self.hand_bone_style)
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(image_rgb_m, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.hand_joint_style, self.hand_bone_style)

            # --- DIBUJAR EL HUD (Texto sobre el video) ---
            image_bgr = cv2.cvtColor(image_rgb_m, cv2.COLOR_RGB2BGR)
            
            # Texto de Atención
            cv2.putText(image_bgr, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
            
            # Texto de Parpadeos
            blink_text = f"Parpadeos: {self.blink_counter}"
            cv2.putText(image_bgr, blink_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.nira_lime, 2, cv2.LINE_AA)
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # --- ENVÍO A LA INTERFAZ ---
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            self.frame_ready.emit(qt_image)

        cap.release()

    def reset_metrics(self):
        """Reinicia los contadores después de enviar una respuesta."""
        self.blink_counter = 0

    def stop(self):
        self.running = False
        self.wait()

    