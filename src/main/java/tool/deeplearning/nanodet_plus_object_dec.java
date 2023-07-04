package tool.deeplearning;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import org.opencv.core.Core;

import java.io.File;
import java.util.Arrays;

/**
*   @desc : NanoDet-Plus目标检测
 *
 *          coco.names：类别
 *          nanodet-plus- 下面是不同尺寸的模型
 *
*   @auth : tyf
*   @date : 2022-05-23  16:40:19
*/
public class nanodet_plus_object_dec {


    // 模型1
    public static OrtEnvironment env;
    public static OrtSession session;

    // 环境初始化
    public static void init(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env = OrtEnvironment.getEnvironment();
        session = env.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型输入-----------");
        session.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型输出-----------");
        session.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });

    }


    public static void main(String[] args) throws Exception{


        // ---------模型输入-----------
        // data -> [1, 3, 320, 320] -> FLOAT
        // ---------模型输出-----------
        // output -> [1, 2125, 112] -> FLOAT
        init(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\nanodet_plus_object_dec\\nanodet-plus-m-1.5x_320.onnx");



    }

}
