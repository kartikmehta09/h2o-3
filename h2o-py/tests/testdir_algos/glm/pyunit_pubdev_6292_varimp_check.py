from builtins import range
import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
import random
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# remember to check varimp for Binomial, Multinomial, Regression at least.

def testvarimp():
    print("Checking variable importance for multinomials....")
    train = h2o.import_file(path=pyunit_utils.locate("smalldata/iris/iris_wheader.csv"))
    myY = "class"
    mX = list(range(1,4))

    model = H2OGeneralizedLinearEstimator(family="multinomial", standardize=True)
    model.train(x=mX, y=myY, training_frame=train)
    print(model.varimp())
    print("NOw what")


if __name__ == "__main__":
  pyunit_utils.standalone_test(testvarimp)
else:
    testvarimp()
