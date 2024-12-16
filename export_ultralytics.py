import argparse

from ultralytics import YOLO

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='yolo.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--dynamic', action='store_true', default=False, help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', default=True, help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=None, help='ONNX: opset version')
    parser.add_argument(
        '--format',
        default='onnx',
        help='include format(s) to export',)
    opt = parser.parse_args()

    return opt

def main(opt):
    model = YOLO(opt.weights, task='detect')
    path = model.export(
        format=opt.format,
        imgsz=opt.imgsz,
        batch=opt.batch_size,
        opset=opt.opset,
        dynamic=opt.dynamic,
        simplify=opt.simplify,
        device=opt.device,
    )

    print(f'Exported: {path}')

if __name__ == '__main__':
    opt = get_args()
    main(opt)

    # e.g. python .\export_ultralytics.py --data ..\manage-dataset\datasets\sugarcane\data.yaml --weights .\runs\train\yolov9-s-sugarcane\weights\best.pt --device 0 --format onnx --imgsz 640
