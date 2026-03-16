from .timeseries_transformer import TimeSeriesTokenizer, TimeSeriesTransformer, TimeSeriesPredictor

model_dict = {
    'timeseries_tokenizer': TimeSeriesTokenizer,
    'timeseries_transformer': TimeSeriesTransformer,
    'timeseries_predictor': TimeSeriesPredictor
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError


