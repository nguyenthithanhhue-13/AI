import importlib
for m in ["sklearn.linear_model", "sklearn.svm", "sklearn.neural_network", "sklearn.feature_extraction.text",
          "sklearn.ensemble", "sklearn.tree", "sklearn.cluster", "scipy.ndimage", "scipy.sparse", "scipy.optimize",
          "scipy.spatial", "torch"]:
    try:
        importlib.import_module(m)
        print("OK     ", m)
    except Exception as e:
        print("BLOCKED", m, "->", str(e)[:110])
