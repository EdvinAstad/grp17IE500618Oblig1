from ai_model import AiModel
from model_type import ModelType

model = (AiModel.assemble()
          .add_meta_data('n_estimators', 150)
          .add_meta_data('random_state', 7)
          .load_features(['Overall Qual', 'Yr Sold', 'Year Remod/Add', 'Year Built', 'Overall Cond'])
          .set_model_type(ModelType.RANDOM_FOREST_REGRESSION)
          .build()
          )
model.print_meta_data()
model.display()