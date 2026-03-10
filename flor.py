from typing import List, Optional, Any
import cv2
import numpy as np
import requests


def run_florence(
    img: np.ndarray, api_url: str = "http://localhost:8000/segment", timeout: int = 120
) -> Optional[List[np.ndarray]]:
    """
    Получает маски зданий из Docker-сервиса Florence-2.

    Аргументы:
        image: путь к файлу (str) или numpy-массив BGR (cv2.imread)
        api_url: адрес эндпоинта сервиса
        timeout: таймаут запроса в секундах
    """

    # 2. Кодируем в bytes (jpeg)
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        print("Не удалось закодировать изображение в jpeg")
        return None

    # 3. Отправляем запрос
    try:
        files = {"file": ("image.jpg", buffer.tobytes(), "image/jpeg")}
        task = "<MORE_DETAILED_CAPTION>"
        text_input = "Analyze the level of violence, aggression and potential danger in the scene. Describe actions (punches, kicks, pushing, weapons?), estimate risk to people: low / medium / high / life-threatening. Mention visible injuries, number of aggressors vs victims, intensity of conflict."

        data = {"task": task, "text_input": text_input, "only_task": True}
        response = requests.post(api_url, files=files, data=data, timeout=timeout)
        response.raise_for_status()

        data = response.json()

        if "error" in data:
            print("Ошибка от сервиса:", data["error"])
            return None

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к сервису: {e}")
        if hasattr(e, "response") and e.response is not None:
            print("Ответ сервера:", e.response.text)
        return None

    # combined_mask = create_masks(data, img)
    return data


text_input = "bricks buildings facades and building walls"
text_input = "the large building in the winter picture"
text_input = "walls of buildings and any metal, concrete and brick structures"
text_input = "all buildings and building facades"
text_input = " Describe in detail what is happening in the scene."
text_input = "the large multi-story residential building covering most of the image"


def create_masks(data, img) -> Any:
    h, w = img.shape[:2]
    # Создаём общую маску для всех зданий
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    n = 0

    for poly_group in data["polygons"]:
        # Временная маска для одного объекта
        obj_mask = np.zeros((h, w), dtype=np.uint8)

        for poly in poly_group:
            if len(poly) < 6:
                continue
            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(obj_mask, [pts], 255)
            n += 1

        # Добавляем объект к общей маске
        combined_mask = cv2.bitwise_or(combined_mask, obj_mask)

    print(f"Получено {n} масок зданий")

    print(f"Создана общая маска зданий")
    return combined_mask
