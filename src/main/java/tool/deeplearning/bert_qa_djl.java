package tool.deeplearning;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
*   @desc : bert 阅读理解（输入段落和问题,给出答案）  ,djl 推理
*   @auth : tyf
*   @date : 2022-06-14  17:17:43
*/
public class bert_qa_djl {

    public static class BertQaInference {


        private BertQaInference() {}

        public static String predict(String paragraph,String question) throws IOException, TranslateException, ModelException {

            QAInput input = new QAInput(question, paragraph);
            System.out.println("Paragraph: {}"+ input.getParagraph());
            System.out.println("Question: {}"+ input.getQuestion());

            Criteria<QAInput, String> criteria =
                    Criteria.builder()
                            .optApplication(Application.NLP.QUESTION_ANSWER)
                            .setTypes(QAInput.class, String.class)
                            .optFilter("backbone", "bert")
                            .optEngine(Engine.getDefaultEngineName())
                            .optProgress(new ProgressBar())
                            .build();

            try (ZooModel<QAInput, String> model = criteria.loadModel()) {
                try (Predictor<QAInput, String> predictor = model.newPredictor()) {
                    return predictor.predict(input);
                }
            }
        }
    }



    public static void main(String[] args) throws IOException, TranslateException, ModelException {


        // 模型会自动下载 mxnet 引擎
        // https://mlrepo.djl.ai/model/nlp/qa/ai/djl/mxnet/bertqa/vocab.json
        // https://mlrepo.djl.ai/model/nlp/question_answer/ai/djl/mxnet/bertqa/0.0.1/static_bert_qa-symbol.json
        // https://mlrepo.djl.ai/model/nlp/question_answer/ai/djl/mxnet/bertqa/0.0.1/static_bert_qa-0002.params.gz

        // 段落
        String paragraph = "BBC Japan was a general entertainment Channel. "
                        + "Which operated between December 2004 and April 2006. "
                        + "It ceased operations after its Japanese distributor folded.";

        // 问题
        String question = "When did BBC Japan start broadcasting?";

        // 答案
        String answer = BertQaInference.predict(paragraph,question);

        System.out.println("答案");
        System.out.println(answer);




    }





}
