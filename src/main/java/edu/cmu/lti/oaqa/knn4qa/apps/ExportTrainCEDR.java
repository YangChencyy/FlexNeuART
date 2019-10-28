package edu.cmu.lti.oaqa.knn4qa.apps;

import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.HashSet;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.knn4qa.cand_providers.LuceneCandidateProvider;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.utils.Const;
import edu.cmu.lti.oaqa.knn4qa.utils.EvalUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.MiscHelper;
import edu.cmu.lti.oaqa.knn4qa.utils.QrelReader;
import edu.cmu.lti.oaqa.knn4qa.utils.StringUtils;

class ExportTrainCEDR extends ExportTrainNegSampleBase {
  
  private static final Logger logger = LoggerFactory.getLogger(ExportTrainCEDR.class);
  
  public static final String FORMAT_NAME = "cedr";
  
  public static final String TEST_RUN_FILE_PARAM = "test_run_file";
  public static final String TEST_RUN_FILE_DESC = "a TREC style test/validation run file";
  
  public static final String DATA_FILE_DOCS_PARAM = "data_file_docs";
  public static final String DATA_FILE_DOCS_DESC = "CEDR data file for docs";

  public static final String DATA_FILE_QUERIES_PARAM = "data_file_queries";
  public static final String DATA_FILE_QUERIES_DESC = "CEDR data file for queries";
  
  public static final String QUERY_DOC_PAIR_FILE_PARAM = "train_pairs_file";
  public static final String QUERY_DOC_PAIR_FILE_DESC = "query-document pairs for training";
  
  
  protected ExportTrainCEDR(LuceneCandidateProvider candProv, ForwardIndex fwdIndex, 
                               QrelReader qrelsTrain, QrelReader qrelsTest) {
    super(candProv, fwdIndex, qrelsTrain, qrelsTest);
  }

//Must be called from ExportTrainBase.addAllOptionDesc
  static void addOptionsDesc(Options opts) {
    opts.addOption(TEST_RUN_FILE_PARAM, null, true, TEST_RUN_FILE_DESC); 
    opts.addOption(DATA_FILE_DOCS_PARAM, null, true, DATA_FILE_DOCS_DESC); 
    opts.addOption(DATA_FILE_QUERIES_PARAM, null, true, DATA_FILE_QUERIES_DESC); 
    opts.addOption(QUERY_DOC_PAIR_FILE_PARAM, null, true, QUERY_DOC_PAIR_FILE_DESC); 
  }

  @Override
  String readAddOptions(CommandLine cmd) {
    String err = super.readAddOptions(cmd);
    if (!err.isEmpty()) {
      return err;
    }
    
    mTestRunFileName = cmd.getOptionValue(TEST_RUN_FILE_PARAM);    
    if (null == mTestRunFileName) {
      return "Specify option: " + TEST_RUN_FILE_PARAM;
    } 
    mDataFileDocsName = cmd.getOptionValue(DATA_FILE_DOCS_PARAM);
    if (null == mDataFileDocsName) {
      return "Specify option: " + DATA_FILE_DOCS_PARAM;
    }
    mDataFileQueriesName = cmd.getOptionValue(DATA_FILE_QUERIES_PARAM);
    if (null == mDataFileQueriesName) {
      return "Specify option: " + DATA_FILE_QUERIES_PARAM;
    } 
    mQueryDocTrainPairsFileName = cmd.getOptionValue(QUERY_DOC_PAIR_FILE_PARAM);
    if (null == mQueryDocTrainPairsFileName) {
      return "Specify option: " + QUERY_DOC_PAIR_FILE_PARAM;
    } 

    return "";
  }
  

  @Override
  void startOutput() throws Exception {
    mOutNum = 0;
    
    mTestRun = MiscHelper.createBufferedFileWriter(mTestRunFileName);
    mDataDocs = MiscHelper.createBufferedFileWriter(mDataFileDocsName);
    mDataQueries = MiscHelper.createBufferedFileWriter(mDataFileQueriesName);
    mQueryDocTrainPairs = MiscHelper.createBufferedFileWriter(mQueryDocTrainPairsFileName);
  }

  @Override
  void finishOutput() throws Exception {
    logger.info("Generated data for " + mOutNum + " queries.");
    
    mTestRun.close();
    mDataDocs.close();
    mDataQueries.close();
    mQueryDocTrainPairs.close();
  }
 
  @Override
  void writeOneEntryData(String queryFieldText, boolean isTestQuery,
                         String queryId,
                         HashSet<String> relDocIds, ArrayList<String> docIds) throws Exception {
    
    if (mSeenQueryIds.contains(queryId)) {
      logger.warn("Ignoring repeating query: " + queryId);
      return;
    }
    mSeenQueryIds.add(queryId);
    
    mDataQueries.write("query\t" + queryId + "\t" + StringUtils.replaceWhiteSpaces(queryFieldText) + Const.NL);
    
    int pos = -1;
    for (String docId : docIds) {
      String text = getDocText(docId);
      
      if (!text.isEmpty()) {
      
        if (!mSeenDocIds.contains(docId)) {
            // documents can sure repeat, but a data file needs to have only one copy
            mDataDocs.write("doc\t" + docId + "\t" + StringUtils.replaceWhiteSpaces(text) + Const.NL);
            mSeenDocIds.add(docId);
        }
      
        ++pos;
        float score = -pos; // the higher is the position, the lower is the "score"
        //Actually I don't even need the relevance flag here
        //int relFlag = relDocIds.contains(docId) ? 1 : 0;
        if (isTestQuery) {
          EvalUtils.saveTrecOneEntry(mTestRun, 
                                     queryId, docId, 
                                     pos, score, Const.FAKE_RUN_ID); 
        } else {
          mQueryDocTrainPairs.write(queryId + "\t" + docId + Const.NL);
        }
      }
    }
  }
  
  int                    mOutNum = 0;
  
  BufferedWriter         mTestRun;
  BufferedWriter         mDataDocs;
  BufferedWriter         mDataQueries;
  BufferedWriter         mQueryDocTrainPairs;
  
  String                 mTestRunFileName;
  String                 mDataFileDocsName;
  String                 mDataFileQueriesName;
  String                 mQueryDocTrainPairsFileName;
  
  HashSet<String>        mSeenDocIds = new HashSet<>();
  HashSet<String>        mSeenQueryIds = new HashSet<>();
  
}