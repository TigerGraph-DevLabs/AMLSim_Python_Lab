!pip install pyTigerGraphBeta
!pip install flat-table
import pyTigerGraphBeta as tg
import pandas as pd
import flat_table
from sklearn import
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

conn = tg.TigerGraphConnection(host="https://aml-sim-sagemaker.i.tgcloud.io", username="tigergraph", password="tigergraph", graphname = "AMLSim")
conn.apiToken = conn.getToken("0dqle85rg436lg25qabtki0lqenunfvj")

if __name == '__main__':
parser = argparser.ArgumentParser()
# SageMaker specific arguments. Defaults are set in the environment variables.

parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

# Save model artifacts
parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

# Train data
parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

args = parser.parse_args()

tx_hop = conn.runInstalledQuery("txMultiHop", {}, timeout="10000000000000000", sizeLimit="1500000000")

df_tx_hop = pd.DataFrame(tx_hop[0]["@@txRecords"])
df_tx_hop = flat_table.normalize(df_tx_hop)

features = ['tx_amount', 's_pagerank', 's_label', 's_min_send_tx', 's_min_receieve_tx', 's_max_send_tx', 's_max_recieve_tx', 's_avg_send_tx', 's_avg_recieve_tx', 's_cnt_recieve_tx', 's_cnt_send_tx', 's_timestamp', 'r_pagerank', 'r_label', 'r_min_send_tx', 'r_min_receieve_tx', 'r_max_send_tx', 'r_max_recieve_tx', 'r_avg_send_tx', 'r_avg_recieve_tx', 'r_cnt_recieve_tx', 'r_cnt_send_tx', 'r_timestamp']

X_train, X_test, y_train, y_test = train_test_split(df_tx_hop[features], df_tx_hop["tx_fraud"], test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))

def model_fn(model_dir):

    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf