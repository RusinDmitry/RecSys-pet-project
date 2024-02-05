from datetime import datetime
from typing import List

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, create_engine


from schema import PostGet, FeedGet, UserGet, Response
import os
import pickle
import pandas as pd
import hashlib

app = FastAPI()

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_features() -> pd.DataFrame:
    return batch_load_sql('SELECT * FROM rusind899_lesson_22')

def get_model_path(path: str, test = 0) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        if test:
            MODEL_PATH = '/workdir/user_input/model_test'
        else:
            MODEL_PATH = '/workdir/user_input/model_control'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models(test = 0):
    model_path = get_model_path("./catboost_model_2.pkl",test = test)
    model = pickle.load(open(model_path, 'rb')) # пример как можно загружать модели
    return model

def text_column_processing(X):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf_idf = TfidfVectorizer()
    text_tf_idf = tf_idf.fit_transform(X['text'])

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    PCA_post_text_dataset = pca.fit_transform(text_tf_idf.toarray())
    PCA_post_text_dataset = pd.DataFrame(PCA_post_text_dataset,
                                         columns=['PCA_text_1', 'PCA_text_2', 'PCA_text_3'],
                                         index=X['post_id'])
    return PCA_post_text_dataset
model_control = load_models()
model_test = load_models(test = 1)
print("Models loaded!")
data_feature = load_features()
df_post = batch_load_sql('SELECT * FROM rusind899_lesson_22_2')
post_data = batch_load_sql('SELECT * FROM public.post_text_df')
print("Feature loaded!")

def conv(df: pd.DataFrame) -> list:
    ls = []
    df.rename(columns={"post_id": "id"}, inplace=True)
    for index, row in df.iterrows():
        ls.append(row.to_dict())
    return ls

def get_exp_group(user_id: int) -> str:
    number = int(hashlib.md5((str(user_id) + 'my_salt').encode()).hexdigest(), 16) % 100
    if number > 50:
        return 'test'
    else:
        return 'control'

@app.get("/post/recommendations/",response_model=Response)
def recommended_posts(id: int, limit: int = 10) -> Response:
    user_tab = data_feature[data_feature['user_id'] == id][['user_id', 'gender', 'age', 'exp_group','country_1.0', 'city_1.0', 'os_iOS',
           'source_organic','userViews','userMeans']]
    df_post['user_id'] = id
    time = datetime.now()
    user_tab['timestamp'] = time
    user_tab['timestamp'] = pd.to_datetime(user_tab['timestamp'])
    user_tab['hour_of_action'] = user_tab['timestamp'].dt.hour
    df_test = pd.merge(
        df_post,
        user_tab,
        on='user_id',
        how='left'
    )
    X = df_test.drop(['user_id', 'timestamp'], axis = 1).rename(columns={"country_1.0": "country_1", "city_1.0": "city_1"})[['post_id', 'hour_of_action', 'gender', 'age', 'exp_group', 'PCA_text_1',
       'PCA_text_2', 'PCA_text_3', 'country_1', 'city_1', 'os_iOS',
       'source_organic', 'topic_covid', 'topic_entertainment', 'topic_movie',
       'topic_politics', 'topic_sport', 'topic_tech', 'userViews',
       'userMeans']]
    exp_group = get_exp_group(df_test['user_id'][0])
    if exp_group == 'control':
        preds = model_control.predict(X)
        y_pred_proba = model_control.predict_proba(X)[:, 1]
    elif exp_group == 'test':
        preds = model_test.predict(X)
        y_pred_proba = model_test.predict_proba(X)[:, 1]
    else:
        raise ValueError('unknown group')

    X['pred'] = preds
    X['pred_proba'] = y_pred_proba
    list_post = X.sort_values(by='pred_proba', ascending=False).head(limit)['post_id'].to_list()
    answer = post_data[post_data['post_id'].isin(list_post)]
    resp = Response(**{'exp_group': exp_group, 'recommendations': conv(answer)})
    return resp


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8899)