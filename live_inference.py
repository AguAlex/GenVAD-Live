import cv2
import numpy as np
import torch

# 1. Încarci modelul antrenat (Student + Teacher)
model.load_state_dict(torch.load("checkpoint-best-student.pth"))
model.eval()

# 2. Te conectezi la camera live (0 pentru webcam, sau un link RTSP de la o cameră IP)
cap = cv2.VideoCapture(0)

# 3. Creezi un "Buffer" (o listă) pentru a ține ultimele 3 cadre (pentru a calcula mișcarea)
cadre_recente = []

while True:
    ret, cadru_curent = cap.read()
    
    # Adăugăm cadrul curent în buffer
    cadre_recente.append(cadru_curent)
    
    # Păstrăm doar ultimele 3 cadre în memorie ca să nu se umple RAM-ul
    if len(cadre_recente) > 3:
        cadre_recente.pop(0)
    
    # 4. Avem nevoie de minim 3 cadre ca să facem diferența (Trecut vs Prezent)
    if len(cadre_recente) == 3:
        cadru_trecut = cadre_recente[0]
        cadru_prezent = cadre_recente[-1]
        
        # 5. CALCULĂM GRADIENTUL LIVE (pe loc)
        gradient = np.abs(cadru_trecut.astype(np.int32) - cadru_prezent.astype(np.int32))
        gradient = gradient.astype(np.uint8)
        
        # Formatăm imaginile pentru model (redimensionare 512x512, normalizare)
        tensor_cadru, tensor_gradient = pregateste_pentru_model(cadru_prezent, gradient)
        
        # 6. TRECEM PRIN MODEL (INFERENȚA)
        with torch.no_grad():
            _, pred_teacher, _, scor_anomalie = model(tensor_cadru, grad_mask=tensor_gradient, ...)
            
        # 7. DECLANȘĂM ALARMA
        # scor_anomalie este un număr. Dacă depășește un prag stabilit de tine (ex. 0.8), e anomalie!
        if scor_anomalie > PRAG_ALERTA:
            cv2.putText(cadru_prezent, "ANOMALIE DETECTATA!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    # Afișăm imaginea pe ecran cu tot cu text
    cv2.imshow("Live GenVAD", cadru_prezent)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break