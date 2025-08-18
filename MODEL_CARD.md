# Model Card — Solar Sage (ResNet18, 2-class)

**Task**: Image classification (CLEAN vs DIRTY)  
**Format**: ONNX (`service/model.onnx`)  
**Labels**: ["CLEAN","DIRTY"]  
**Normalization**: IMAGENET  
**Threshold**: 0.40 (best F1 on validation)

## Metrics (validation @ 0.40)
- Accuracy ≈ 0.88
- Precision ≈ 0.87
- Recall ≈ 0.83
- F1 ≈ 0.85

## Provenance
- Trained locally; exported to ONNX.
- SHA256(onnx): 70058eb62be9ef59a2eeefcfd598d9be8e79a03bfd4baec1d7bfb2560b49de0b
- Export: PyTorch → ONNX (ResNet18)

## Notes / Limits
- Sensitive to lighting/angle/occlusion; validate on your data.
