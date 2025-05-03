import joblib
obj = joblib.load("app/models/url_adv_pipeline_fast.pkl")
print("Loaded object type:", type(obj))
print("Number of elements inside:", len(obj))
print("Contents:", obj)
