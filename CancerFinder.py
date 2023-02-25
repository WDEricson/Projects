{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN+/NQjBDbOZGLa3KupWY5y",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/WDEricson/Projects/blob/Weather/CancerFinder.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "T1n-7zm3aaVp",
        "outputId": "6e5e7046-5a1d-4c73-c6e4-13f967ba6dce"
      },
      "source": [
        "import random\n",
        "#define cancer in terms of the array\n",
        "def cancer(array):\n",
        "#set the initial values of both G and B to zero\n",
        "  G=0\n",
        "  B=0\n",
        "#for loop to generate beginning syntax\n",
        "  for i in range(20):\n",
        "    print(\"    [\", end=\" \")\n",
        "    #for loop with imbedded if statements to transform \n",
        "    #any number greate than 7 to the letter B\n",
        "    #and any number less than 3 to the letter G\n",
        "    for j in range(30):\n",
        "      if array[i][j]>=8:\n",
        "        array[i][j]='B'\n",
        "        #add one to the counter for Bad cells\n",
        "        B+=1\n",
        "      elif array[i][j]<=2:\n",
        "        array[i][j]='G'\n",
        "        #add one to the counter for Good Cells\n",
        "        G+=1\n",
        "        #if the number is between 2 and 8, print a blank space\n",
        "      else:\n",
        "        array[i][j]=' '\n",
        "        #synax and new line\n",
        "      print(array[i][j], end=\" \")\n",
        "    print(']')\n",
        "  #space down\n",
        "  print(\"\\n\"*2)\n",
        "  #display the resulting counts for the good and bad cells\n",
        "  print(G,\" Good Cells\",\"&\",B,' Bad Cells')\n",
        "array = [[random.randint(1,9) for c in range(30)] for r in range(20)]\n",
        "#call the defined function\n",
        "cancer(array)\n"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "    [ B G B G     B     G G B     B       G   G             G     ]\n",
            "    [ B B         B   B   B     B   B     B   B B     G   G   G B ]\n",
            "    [   G     B       B         G     B       B   B     B   B   B ]\n",
            "    [     B   B   B B G       B   G B     B     G       B   B     ]\n",
            "    [ B G         B G B G     G G     B         B     B   B B     ]\n",
            "    [ G B     B   B           G   G B   B     B G B B     G   G B ]\n",
            "    [   B B   G         B               B   B B G     G       G   ]\n",
            "    [   B         B B       B   G G G B G G G B   B     B   B     ]\n",
            "    [ B G             B   B   B         G B G         B   B       ]\n",
            "    [   G   B           G B G         B G B               B B G G ]\n",
            "    [     G G G       G B       B         B G B B         B G B   ]\n",
            "    [           B       B   G B   G G               G B     G   G ]\n",
            "    [           G B   B G       B   G B B     B B         G B B   ]\n",
            "    [ G   G     G   G   G B B G   B   G       B     B     B G     ]\n",
            "    [     G G B B G   B     G B         B               B B G B G ]\n",
            "    [   G   G     G G G B   B         B   G G   B               B ]\n",
            "    [       B   B G B G   G G B     B G   G     B B       B       ]\n",
            "    [     B         B G B B G     G   G B   B G G     B G G B   B ]\n",
            "    [     G         G G G       G B                 G     G       ]\n",
            "    [     B     G     B         G B B   G B   B     G         G   ]\n",
            "\n",
            "\n",
            "\n",
            "108  Good Cells & 141  Bad Cells\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "l8tX1ud2fDm5",
        "outputId": "25a24522-7843-4e74-87e1-c2af91f6fb71"
      },
      "source": [
        "import random\n",
        "for j in range(1,11):\n",
        "  print()\n",
        "  for i in range(1,7):\n",
        "    print(random.randint(1,9), end='')"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "183845\n",
            "942385\n",
            "641383\n",
            "879614\n",
            "312455\n",
            "684682\n",
            "661437\n",
            "542313\n",
            "394717\n",
            "148231"
          ]
        }
      ]
    }
  ]
}