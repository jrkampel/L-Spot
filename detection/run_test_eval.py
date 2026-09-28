
from ultralytics import YOLO

base = '.'

for exp in ['exp1', 'exp2', 'exp3']:
    print("\n=== Testing " + exp + " on full-slice test set ===")
    model = YOLO(base + '/runs/detect/yolov8m_' + exp + '/weights/best.pt')
    results = model.val(
        data=base + '/dataset.yaml',
        split='test',
        imgsz=512,
        batch=16,
        project=base + '/runs/detect',
        name='yolov8m_' + exp + '_testeval',
        exist_ok=True
    )
    print("Precision: " + str(round(results.box.mp, 3)))
    print("Recall:    " + str(round(results.box.mr, 3)))
    print("mAP50:     " + str(round(results.box.map50, 3)))
    print("mAP50-95:  " + str(round(results.box.map, 3)))
