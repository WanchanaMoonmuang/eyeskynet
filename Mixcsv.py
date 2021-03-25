import pandas as pd
import os

def main() :
    Dir = "D:\Downloads\eyeskynet\Output"
    os.chdir(Dir)
    name1 = 'H_test.csv'
    name2 = 'O_test.csv'
    mix_data2(name1,name2)


def mix_data2(name1,name2):

    dfH = pd.read_csv(name1)
    dfG = pd.read_csv(name2)
    
    mix = pd.concat([dfH,dfG])
    mix.drop(mix.filter(regex="Unname"),axis=1, inplace=True)
    mix.to_csv('mixed_data.csv',index = False)


def mix_data3(name1,name2,name3):

    dfH = pd.read_csv(name1)
    dfG = pd.read_csv(name2)
    dfO = pd.read_csv(name3)

    mix = pd.concat([dfH,dfG,dfO])
    mix.drop(mix.filter(regex="Unname"),axis=1, inplace=True)

    mix.to_csv('mixed_data.csv',index = False)



if __name__ == "__main__":
    main()
