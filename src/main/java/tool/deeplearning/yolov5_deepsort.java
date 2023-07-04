package tool.deeplearning;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import org.opencv.core.Core;

import java.io.File;
import java.util.Arrays;

/**
*   @desc : 目标跟踪 deepsort + yolov5
 *
*   @auth : tyf
*   @date : 2022-05-15  17:23:30
*/
public class yolov5_deepsort {


    // 模型1
    public static OrtEnvironment env1;
    public static OrtSession session1;

    // 模型2
    public static OrtEnvironment env2;
    public static OrtSession session2;


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


    public static void main(String[] args) throws Exception{

        // deepsort
        // ---------模型1输入-----------
        // input_1 -> [-1, 3, 64, 128] -> FLOAT
        // ---------模型1输出-----------
        // output_1 -> [-1, 128] -> FLOAT
        init1(new File("").getCanonicalPath()+"\\model\\deeplearning\\yolov5_deepsort\\deepsort.onnx");


        // yolov5s
        // ---------模型2输入-----------
        // input0 -> [-1, 3, 640, 640] -> FLOAT
        // ---------模型2输出-----------
        // output -> [-1, 25200, 6] -> FLOAT
        // 1016 -> [1, 3, 80, 80, 6] -> FLOAT
        // 1337 -> [1, 3, 40, 40, 6] -> FLOAT
        // 1658 -> [1, 3, 20, 20, 6] -> FLOAT
        init2(new File("").getCanonicalPath()+"\\model\\deeplearning\\yolov5_deepsort\\yolov5s.onnx");




    }


}
