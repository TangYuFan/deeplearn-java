package tool.deeplearning;


import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import org.opencv.core.Core;

import java.io.File;
import java.util.Arrays;

/**
 *  @Desc: stable_diffusion AI画图，支持图生图、文生图
 *  @Date: 2022-05-13 18:46:38
 *  @auth: TYF
 */
public class stable_diffusion_onnx {


    // 模型1
    public static OrtEnvironment env1;
    public static OrtSession session1;

    // 模型2
    public static OrtEnvironment env2;
    public static OrtSession session2;

    // 模型3
    public static OrtEnvironment env3;
    public static OrtSession session3;

    // 环境初始化
    public static void init1(String weight) throws Exception{

        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env1 = OrtEnvironment.getEnvironment();
        session1 = env1.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型1输入-----------");
        session1.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型1输出-----------");
        session1.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
//        session1.getMetadata().getCustomMetadata().entrySet().forEach(n->{
//            System.out.println("元数据:"+n.getKey()+","+n.getValue());
//        });

    }



    // 环境初始化
    public static void init2(String weight) throws Exception{

        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env2 = OrtEnvironment.getEnvironment();
        session2 = env2.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型2输入-----------");
        session2.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型2输出-----------");
        session2.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session2.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });

    }



    // 环境初始化
    public static void init3(String weight) throws Exception{

        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env3 = OrtEnvironment.getEnvironment();
        session3 = env3.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型3输入-----------");
        session3.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型3输出-----------");
        session3.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
//        session3.getMetadata().getCustomMetadata().entrySet().forEach(n->{
//            System.out.println("元数据:"+n.getKey()+","+n.getValue());
//        });

    }


    public static class SdWorker{

        // 正面提示词
        String text1;
        // 负面提示词
        String text2;
        public SdWorker(String text1,String text2) {
            this.text1 = text1;
            this.text2 = text2;
        }
        public void clip(){

            // ---------模型2输入-----------
            // onnx::Reshape_0 -> [1, 77] -> INT64
            // ---------模型2输出-----------
            // 2271 -> [1, 77, 768] -> FLOAT

        }
        public void diffusion(){

        }
        public void encoder(){

        }
        public void show(){

        }
    }

    public static void main(String[] args) throws Exception{

        // ---------模型1输入-----------
        // input.1 -> [1, 4, 64, 64] -> FLOAT
        // ---------模型1输出-----------
        // 815 -> [1, 3, 512, 512] -> FLOAT
        init1(new File("").getCanonicalPath()+"\\model\\deeplearning\\stable_diffusion_onnx\\AutoencoderKL-fp32.onnx");

        // ---------模型2输入-----------
        // onnx::Reshape_0 -> [1, 77] -> INT64
        // ---------模型2输出-----------
        // 2271 -> [1, 77, 768] -> FLOAT
        init2(new File("").getCanonicalPath()+"\\model\\deeplearning\\stable_diffusion_onnx\\FrozenCLIPEmbedder-fp32.onnx");

        // ---------模型3输入-----------
        // x -> [1, 4, 64, 64] -> FLOAT
        // t -> [1] -> FLOAT
        // cc -> [1, -1, 768] -> FLOAT
        // ---------模型3输出-----------
        // out -> [1, 4, 64, 64] -> FLOAT
        init3(new File("").getCanonicalPath()+"\\model\\deeplearning\\stable_diffusion_onnx\\UNetModel-fp16.onnx");

        // prompt
        SdWorker worker = new SdWorker(
                "a beautiful girl as an enchanting forest elf sitting on a tree, " +
                "serene expression, wearing a flowing green dress with intricate details",
                null
        );

        // text encode
        worker.clip();

        // 扩散
        worker.diffusion();
//        worker.encoder();
        worker.show();


    }

}
