from io_utils import ComputationalModels

#res = ComputationalModels().compare_models()
res = ComputationalModels().compare_models(tsne=True)
for k, v in res.items():
    print([k, v])
import pdb; pdb.set_trace()
