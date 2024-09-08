import tensorrt as trt
import onnx
import pycuda.driver as cuda
import pycuda.autoinit

# ONNXモデルをロード
onnx_model_path = "./best-end2end.onnx"
onnx_model = onnx.load(onnx_model_path)

# TensorRTのloggerを作成
logger = trt.Logger(trt.Logger.WARNING)

# TensorRT builder, network, parserの作成
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# ONNXモデルをTensorRTネットワークにパース
with open(onnx_model_path, "rb") as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

# TensorRTエンジンをビルド
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GBのワークスペース
engine = builder.build_engine(network, config)

# エンジンを保存
with open("model.plan", "wb") as f:
    f.write(engine.serialize())
