/**
 * Created by ellenwong on 5/8/16.
 */
object SimpleMLApp {

  def main(args: Array[String]) = {
    println("Simple Loan MLApp")

    //DataSource: https://www.kaggle.com/wendykan/lending-club-loan-data
    /*
    * Problem of interest:
    * Can we use the loan data from lending club to predict whether or not someone will be late on their loan?
    * How do we define "late"?
    * */

    //Load Data (csv)

    //Split data into training, validation (Is the data temporal??)
    //http://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validation

    //Extract Features from dataset?
    /*
    * (1) age
    * (2) Sex
    * (3) City/State/county
    * (4) Loan amount
    * (5) Has been late before
    * (6) less than x days before due
    * (7) ???
    * */

    //Extract golden label
    /*
    * Late = ???
    * */

    // Pick algorithm, but start with logistic regression
    /*
    * (1) logistic regression?
    * (2) linear regression?
    * (Bonus) what others?? (SVM? decision tree? daboast? Do some research
    * */

    //


  }
}
