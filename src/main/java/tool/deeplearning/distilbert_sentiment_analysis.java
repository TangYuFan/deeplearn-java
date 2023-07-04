package tool.deeplearning;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;

/**
*   @desc : DistilBERT 情感分析（英文文本），djl 推理
*   @auth : tyf
*   @date : 2022-06-14  09:58:21
*/
public class distilbert_sentiment_analysis {


    public static class PtDistilBertTranslator implements Translator<String, Classifications> {

        private Vocabulary vocabulary;
        private BertTokenizer tokenizer;

        /** {@inheritDoc} */
        @Override
        public void prepare(TranslatorContext ctx) throws IOException {
            Model model = ctx.getModel();
            URL url = model.getArtifact("distilbert-base-uncased-finetuned-sst-2-english-vocab.txt");
            vocabulary =
                    DefaultVocabulary.builder().addFromTextFile(url).optUnknownToken("[UNK]").build();
            tokenizer = new BertTokenizer();
        }

        /** {@inheritDoc} */
        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            NDArray raw = list.singletonOrThrow();
            NDArray computed = raw.exp().div(raw.exp().sum(new int[] {0}, true));
            return new Classifications(Arrays.asList("Negative", "Positive"), computed);
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            List<String> tokens = tokenizer.tokenize(input);
            long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();
            long[] attentionMask = new long[tokens.size()];
            Arrays.fill(attentionMask, 1);
            NDManager manager = ctx.getNDManager();
            NDArray indicesArray = manager.create(indices);
            NDArray attentionMaskArray = manager.create(attentionMask);
            return new NDList(indicesArray, attentionMaskArray);
        }
    }

    public static class SentimentAnalysis {

        private SentimentAnalysis() {
        }


        public static Classifications predict()
                throws MalformedModelException, ModelNotFoundException, IOException,
                TranslateException {
            String input = "I like DJL. DJL is the best DL framework!";

            System.out.println("输入:");
            System.out.println(input);


            Criteria<String, Classifications> criteria =
                    Criteria.builder()
                            .optApplication(Application.NLP.SENTIMENT_ANALYSIS)
                            .setTypes(String.class, Classifications.class)
                            // This model was traced on CPU and can only run on CPU
                            .optDevice(Device.cpu())
                            .optProgress(new ProgressBar())
                            .build();

            ZooModel<String, Classifications> model = criteria.loadModel();

            System.out.println("model:"+model.getTranslator().getClass());

            Predictor<String, Classifications> predictor = model.newPredictor();
            return predictor.predict(input);
        }
    }



    public static void main(String[] args) throws IOException, TranslateException, ModelException {

        // 模型自动下载
        // https://mlrepo.djl.ai/model/nlp/sentiment_analysis/ai/djl/pytorch/distilbert/traced_distilbert_sentiment/0.0.1/distilbert-base-uncased-finetuned-sst-2-english-vocab.txt.gz
        // https://mlrepo.djl.ai/model/nlp/sentiment_analysis/ai/djl/pytorch/distilbert/traced_distilbert_sentiment/0.0.1/traced_distilbert_sst_english.pt.gz


        // 默认 translater
        // ai.djl.pytorch.zoo.nlp.sentimentanalysis.PtDistilBertTranslator
        Classifications classifications = SentimentAnalysis.predict();

        System.out.println("推理结果:");
        System.out.println(classifications);
    }



}
