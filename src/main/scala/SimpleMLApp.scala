import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.classification

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



    /*Data
    *
    id,member_id,loan_amnt,funded_amnt,funded_amnt_inv,term,int_rate,installment,grade,sub_grade,emp_title,emp_length,home_ownership,annual_inc,verification_status,issue_d,loan_status,pymnt_plan,url,desc,purpose,title,zip_code,addr_state,dti,delinq_2yrs,earliest_cr_line,inq_last_6mths,mths_since_last_delinq,mths_since_last_record,open_acc,pub_rec,revol_bal,revol_util,total_acc,initial_list_status,out_prncp,out_prncp_inv,total_pymnt,total_pymnt_inv,total_rec_prncp,total_rec_int,total_rec_late_fee,recoveries,collection_recovery_fee,last_pymnt_d,last_pymnt_amnt,next_pymnt_d,last_credit_pull_d,collections_12_mths_ex_med,mths_since_last_major_derog,policy_code,application_type,annual_inc_joint,dti_joint,verification_status_joint,acc_now_delinq,tot_coll_amt,tot_cur_bal,open_acc_6m,open_il_6m,open_il_12m,open_il_24m,mths_since_rcnt_il,total_bal_il,il_util,open_rv_12m,open_rv_24m,max_bal_bc,all_util,total_rev_hi_lim,inq_fi,total_cu_tl,inq_last_12m
1077501,1296599,5000.0,5000.0,4975.0, 36 months,10.65,162.87,B,B2,,10+ years,RENT,24000.0,Verified,Dec-2011,Fully Paid,n,https://www.lendingclub.com/browse/loanDetail.action?loan_id=1077501,  Borrower added on 12/22/11 > I need to upgrade my business technologies.<br>,credit_card,Computer,860xx,AZ,27.65,0.0,Jan-1985,1.0,,,3.0,0.0,13648.0,83.7,9.0,f,0.0,0.0,5861.07141425,5831.78,5000.0,861.07,0.0,0.0,0.0,Jan-2015,171.62,,Jan-2016,0.0,,1.0,INDIVIDUAL,,,,0.0,,,,,,,,,,,,,,,,,
1077430,1314167,2500.0,2500.0,2500.0, 60 months,15.27,59.83,C,C4,Ryder,< 1 year,RENT,30000.0,Source Verified,Dec-2011,Charged Off,n,https://www.lendingclub.com/browse/loanDetail.action?loan_id=1077430,  Borrower added on 12/22/11 > I plan to use this money to finance the motorcycle i am looking at. I plan to have it paid off as soon as possible/when i sell my old bike. I only need this money because the deal im looking at is to good to pass up.<br><br>  Borrower added on 12/22/11 > I plan to use this money to finance the motorcycle i am looking at. I plan to have it paid off as soon as possible/when i sell my old bike.I only need this money because the deal im looking at is to good to pass up. I have finished college with an associates degree in business and its takingmeplaces<br>,car,bike,309xx,GA,1.0,0.0,Apr-1999,5.0,,,3.0,0.0,1687.0,9.4,4.0,f,0.0,0.0,1008.71,1008.71,456.46,435.17,0.0,117.08,1.11,Apr-2013,119.66,,Sep-2013,0.0,,1.0,INDIVIDUAL,,,,0.0,,,,,,,,,,,,,,,,,

    * */

    val dataBasePath = "/Users/ellenwong/Desktop/LoanCSVs/"
    val inputFileName = "loanSample.csv"
    val inputFile = dataBasePath + inputFileName

    //Read the raw file
    val conf = new SparkConf().setAppName("LoanInsightApp").setMaster("local")
    val sc = new SparkContext(conf)
    val rawLoanData: RDD[String] = sc.textFile(inputFile)
    println(s"\nrawLoanData loaded: ${rawLoanData.count()} ")
    val loanSimpleData: RDD[LoanSimple] = rawLoanData.map {
      LoanSimple(_)
    }
    println(s"\nloanSimpleData loaded: ${loanSimpleData.count()}\n")

    val goldenlabelFcn: (LoanSimple) => Double = (loan: LoanSimple) => if(loan.loan_status.toLowerCase.contains("late")) 1.0 else 0.0
    // Extract a Feature
    val loanFeatureExtractor = new LoanAmountFeatureExtractor
    val extractedLabelPoint = loanSimpleData.map{ loanSimpleData =>
      LabeledPoint(goldenlabelFcn(loanSimpleData),
        Vectors.dense(loanFeatureExtractor.extractDouble(loanSimpleData)))
    }
    println(s"\nExtractedLoanFeatures: \n${extractedLabelPoint.collect().mkString("\n")}\n")

    //TODO: Create FeatureMap

    //TODO: Splitt data

    //Generate a Model

    val model: LogisticRegressionModel = new LogisticRegressionWithLBFGS().run(extractedLabelPoint)

    //TODO: replace with actual vector instead of vector from training data
    val loanVector: RDD[Vector] = loanSimpleData.map(loan => Vectors.dense(loanFeatureExtractor.extractDouble(loan)))
    val results: RDD[Double] = model.predict(loanVector)

    println(s"\nModel prediction results: \n${results.collect().mkString("\n")}\n")

    sc.stop()
  }


  class LoanAmountFeatureExtractor extends Serializable{
    val featureName = "loanAmount"
    def extractDouble(loan: LoanSimple) = {
      loan.loan_amnt.toDouble
    }
  }


  case class LoanSimple(id:String, loan_amnt: String, loan_status: String)
  object LoanSimple {
    def apply(rawInput: String) = {
      val splittedInput: Array[String] = rawInput.split(",")
      new LoanSimple(id = splittedInput(0), loan_amnt = splittedInput(2), loan_status = splittedInput(16))
    }
  }

  case class Loan(id: String ,member_id: String,
                       loan_amnt: String,
                       funded_amnt: String,
                       funded_amnt_inv: String,
                       term: String,
                       int_rate: String,
                       installment: String,
                       grade: String,
                       sub_grade: String,
                       emp_title: String,
                       emp_length: String,
                       home_ownership: String,
                       annual_inc: String,
                       verification_status: String,
                       issue_d: String,
                       loan_status: String,
                       pymnt_plan: String,
                       url: String,
                       desc: String,
                       purpose: String,
                       title: String,
                       zip_code: String,
                       addr_state: String,
                       dti: String,
                       delinq_2yrs: String,
                       earliest_cr_line: String,
                       inq_last_6mths: String,
                       mths_since_last_delinq: String,
                       mths_since_last_record: String,
                       open_acc: String,
                       pub_rec: String,
                       revol_bal: String,
                       revol_util: String,
                       total_acc: String,
                       initial_list_status: String,
                       out_prncp: String,
                       out_prncp_inv: String,
                       total_pymnt: String,
                       total_pymnt_inv: String,
                       total_rec_prncp: String,
                       total_rec_int: String,
                       total_rec_late_fee: String,
                       recoveries: String,
                       collection_recovery_fee: String,
                       last_pymnt_d: String,
                       last_pymnt_amnt: String,
                       next_pymnt_d: String,
                       last_credit_pull_d: String,
                       collections_12_mths_ex_med: String,
                       mths_since_last_major_derog: String,
                       policy_code: String,
                       application_type: String,
                       annual_inc_joint: String,
                       dti_joint: String,
                       verification_status_joint: String,
                       acc_now_delinq: String,
                       tot_coll_amt: String,
                       tot_cur_bal: String,
                       open_acc_6m: String,
                       open_il_6m: String,
                       open_il_12m: String,
                       open_il_24m: String,
                       mths_since_rcnt_il: String,
                       total_bal_il: String,
                       il_util: String,
                       open_rv_12m: String,
                       open_rv_24m: String,
                       max_bal_bc: String,
                       all_util: String,
                       total_rev_hi_lim: String,
                       inq_fi: String,
                       total_cu_tl: String,
                       inq_last_12m: String
                        )

  object Loan {
//    def apply(loanRawInfo: Array[String]): Loan = {
//      Loan()
//    }
  }

}
