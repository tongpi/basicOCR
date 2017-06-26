package com.gds;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.seg.common.Term;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;

import java.io.*;
import java.util.*;


/**
 * Created by yangzhanku on 2017/6/5.
 */
public class Test {

    public static void testBuildAndParseWithBigFile(String inputTextPath, String outputTextPath) {

        List<String> text = loadText(inputTextPath);
        BufferedWriter bw = null;
        try {
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputTextPath), "utf-8"));
            for (String line :
                    text) {
                List<Term> lists = HanLP.segment(line);
                for (Term term :
                        lists) {
                    try {
                        bw.write(term.word.trim() + " ");
                        bw.flush();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

            }

        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } finally {
            try {
                bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }

    private static List<String> loadText(String path) {
        ArrayList<String> tm = getDataPath();
        // Load test data from disk
        Set<String> dictionary = loadDictionary(tm.get(0));
        // You can use any type of Map to hold data
        TreeMap<String, String> map = new TreeMap<String, String>();
        for (String key : dictionary) {
            map.put(key, key);
        }
        // Build an AhoCorasickDoubleArrayTrie
        AhoCorasickDoubleArrayTrie<String> acdat = new AhoCorasickDoubleArrayTrie<String>();
        acdat.build(map);
        // Test it

        List<String> list = new ArrayList<String>();
        File file = new File(path);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while ((line = br.readLine()) != null) {
                List<AhoCorasickDoubleArrayTrie<String>.Hit<String>> wordList = acdat.parseText(line);
                for (AhoCorasickDoubleArrayTrie<String>.Hit<String> ah :
                        wordList) {
                    line = line.replace(ah.value.toString(), "");
                }
                list.add(line + "\n");
            }
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                e.getMessage();
            }
        }

        return list;
    }


    private static Set<String> loadDictionary(String path) {
        Set<String> dictionary = new TreeSet<String>();
        File file = new File(path);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while ((line = br.readLine()) != null) {
                dictionary.add(line);
            }
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.getMessage();
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return dictionary;
    }

    public static ArrayList<String> getDataPath() {
        ArrayList<String> arraylist = new ArrayList<String>();
        SAXReader reader = new SAXReader();
        try {
            Document document = reader.read(new File("src/main/resources/dataPath.xml"));
            Element dataStore = document.getRootElement();
            Iterator it = dataStore.elementIterator();
            while (it.hasNext()) {
                Element node = (Element) it.next();
                //解析子节点的信息
                Iterator iter = node.elementIterator();
                while (iter.hasNext()) {
                    Element dataChild = (Element) iter.next();
//                    System.out.println("节点名：" + dataChild.getName() + "--节点值：" + dataChild.getStringValue());
                    arraylist.add(dataChild.getStringValue());
                }
            }
        } catch (DocumentException e) {
            e.printStackTrace();
        }
        return arraylist;
    }

    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        ArrayList<String> tm = getDataPath();
        testBuildAndParseWithBigFile(tm.get(1), tm.get(2));
        long end = System.currentTimeMillis();
        System.out.println("总共耗时：" + (end - start));
    }
}
