import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Test {
    public static void main(String[] args){
        callpython("D:/yolov5-v7.0/datasets/light/images/val/IMG_20230803_145345.jpg", "detect_light.py");
    }

    public static void callpython(String path, String py){
        try {
            System.out.println("start");
            String rpath = path;
            Process pr=Runtime.getRuntime().exec("D:\\anaconda\\Scripts\\activate.bat && conda activate pytorch && python D:\\yolov5-v7.0\\"+py+" "+rpath);
            BufferedReader in = new BufferedReader(new InputStreamReader(
                    pr.getInputStream(),"GBK"));
            String inline = null;
            while ((inline = in.readLine()) != null) {
                System.out.println(inline);
            }
            in.close();
            int re = pr.waitFor();
            System.out.println(re);
            System.out.println("end");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
