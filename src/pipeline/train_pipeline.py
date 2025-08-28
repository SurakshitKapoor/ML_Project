

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

obj=DataIngestion()
train_data,test_data=obj.initiate_data_ingestion()

data_transformation=DataTransformation()
train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

model_trainer = ModelTrainer()
best_model, r2_sq_score = model_trainer.initiate_model_trainer(train_arr,test_arr)

print(f"""
      The Best Model is : {best_model}; and 
      The Best R2_Square Score is : {r2_sq_score}
      
      """)

if __name__ == "__main__":
    print("Starting from the \'train_pipeline.py\' file ")
