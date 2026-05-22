import torch # type: ignore

def load_quant_info_from_checkpoint(path: str):
    """
    Loads the new (copied) quantized checkpoint and returns (quant_info, num_bits, scales_dict).
    quant_info: { name: (q_tensor, scale) }
    """
    payload = torch.load(path, map_location="cpu")
    if not (isinstance(payload, dict) and "qstate_dict" in payload and "meta" in payload and "scales" in payload["meta"]):
        raise ValueError("Checkpoint missing qstate_dict/meta/scales.")
    quant_info = payload["qstate_dict"]
    scales = payload["meta"]["scales"]
    num_bits = payload["meta"].get("num_bits", None)

    return quant_info, num_bits, scales

def clone_quant_info(quant_info: dict):
    editable = {}
    for name, (q, s) in quant_info.items():
        # clone() to decouple from the loaded tensors; keep dtype/device
        editable[name] = (q.clone(), s)
    return editable

def revert_back_some_layers(editable: dict, original:dict, select_layers: list):
    TotalCount = 0
    SkippedCount = 0

    keys = list(editable.keys())

    #verify
    for idx in range(len(editable)):
        k = keys[idx]
        _, s1 = original[k]
        _, s2 = original[k]
        assert s1==s2, "scales not same"

    for idx in range(len(editable)):
        k = keys[idx]
        if(idx in select_layers):
            # editable[k] = deepcopy(original[k])
            # editable[k] = original[k]
            t, s = original[k]
            editable[k] = (t.clone(), s)
            SkippedCount += editable[k][0].numel()
        TotalCount += editable[k][0].numel()

    return editable, TotalCount, SkippedCount