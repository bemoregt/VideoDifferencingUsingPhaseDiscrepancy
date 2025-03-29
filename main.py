import cv2
import numpy as np
import imageio
from pathlib import Path

def complex_numpy(real, imag):
    return real * np.exp(1j * imag)

def compute_phase_discrepancy(frame1, frame2):
    d1 = np.fft.fft2(frame1)
    d2 = np.fft.fft2(frame2)
    phase1 = np.angle(d1)
    amp1 = np.abs(d1)
    phase2 = np.angle(d2)
    amp2 = np.abs(d2)
    z1 = complex_numpy((amp1 - amp2), phase1)
    z2 = complex_numpy((amp2 - amp1), phase2)
    m1 = np.fft.ifft2(z1)
    m2 = np.fft.ifft2(z2)
    m11 = np.abs(m1)
    m22 = np.abs(m2)
    m12 = np.multiply(m11, m22)
    result = np.interp(m12, (m12.min(), m12.max()), (0, 255))
    return result

def compute_subtract(frame1, frame2):
    return cv2.absdiff(frame1, frame2)

def concatenate_images(img1, img2):
    return np.concatenate((img1, img2), axis=1)

def process_video(input_video, output_gif):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video}")
        return
    
    # 비디오 프레임 속성 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 결과 GIF 작성자 설정
    writer = imageio.get_writer(output_gif, fps=fps)
    
    prev_frame = None
    success, frame = cap.read()
    
    while success:
        # 그레이스케일로 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            discrepancy = compute_phase_discrepancy(prev_frame, gray_frame)
            subtract = compute_subtract(prev_frame, gray_frame)
            result = concatenate_images(subtract, discrepancy.astype(np.uint8))
            writer.append_data(cv2.cvtColor(result, cv2.COLOR_GRAY2RGB))
        
        prev_frame = gray_frame
        success, frame = cap.read()
    
    # 자원 해제
    cap.release()
    writer.close()
    print(f"Processing complete. Output saved to {output_gif}")

# 사용 예시
if __name__ == "__main__":
    input_path = 'path/to/your/input/video.mp4'
    output_path = 'path/to/your/output/result.gif'
    process_video(input_path, output_path)