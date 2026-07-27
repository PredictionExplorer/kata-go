#include "../tests/tests.h"

#include "../neuralnet/nninputs.h"
#include "../program/setup.h"
#include "../search/searchparams.h"

using namespace std;
using namespace TestCommon;

namespace {
  static double slowScoreMaximizingUtility(double score, double power, double scale) {
    if(score == 0.0)
      return 0.0;
    double magnitude = pow(1.0 + std::fabs(score) / scale,power) - 1.0;
    return score > 0.0 ? magnitude : -magnitude;
  }

  static double slowExpectedScoreMaximizingUtility(
    double scoreMean,
    double scoreStdev,
    double power,
    double scale,
    double lowerBound,
    double upperBound
  ) {
    if(scoreStdev <= 0.0) {
      double score = std::min(upperBound,std::max(lowerBound,scoreMean));
      return slowScoreMaximizingUtility(score,power,scale);
    }

    static const double sqrtTwo = 1.41421356237309504880;
    static const double invSqrtTwoPi = 0.39894228040143267794;
    auto normalCDF = [](double x) {
      return 0.5 * erfc(-x / sqrtTwo);
    };
    auto normalSurvival = [](double x) {
      return 0.5 * erfc(x / sqrtTwo);
    };
    auto integrand = [&](double z) {
      double score = std::min(upperBound,std::max(lowerBound,scoreMean + scoreStdev*z));
      return slowScoreMaximizingUtility(score,power,scale) * invSqrtTwoPi * exp(-0.5*z*z);
    };

    double lowerZ = (lowerBound-scoreMean) / scoreStdev;
    double upperZ = (upperBound-scoreMean) / scoreStdev;
    double result =
      normalCDF(lowerZ) * slowScoreMaximizingUtility(lowerBound,power,scale) +
      normalSurvival(upperZ) * slowScoreMaximizingUtility(upperBound,power,scale);

    double integrationLower = std::max(lowerZ,-12.0);
    double integrationUpper = std::min(upperZ,12.0);
    if(integrationLower >= integrationUpper)
      return result;

    const int numSteps = 100000;
    double step = (integrationUpper-integrationLower) / numSteps;
    double sum = integrand(integrationLower) + integrand(integrationUpper);
    for(int i = 1; i<numSteps; i++)
      sum += (i % 2 == 0 ? 2.0 : 4.0) * integrand(integrationLower + i*step);
    return result + sum * step / 3.0;
  }
}

void Tests::runScoreTests() {
  cout << "Running score and utility tests" << endl;
  ostringstream out;

  auto printScoreStats = [&out](const Board& board, const BoardHistory& hist) {
    out << "Black self komi wins/draw=0.5: " << hist.currentSelfKomi(P_BLACK, 0.5) << endl;
    out << "White self komi wins/draw=0.5: " << hist.currentSelfKomi(P_WHITE, 0.5) << endl;
    out << "Black self komi wins/draw=0.25: " << hist.currentSelfKomi(P_BLACK, 0.25) << endl;
    out << "White self komi wins/draw=0.25: " << hist.currentSelfKomi(P_WHITE, 0.25) << endl;
    out << "Black self komi wins/draw=0.75: " << hist.currentSelfKomi(P_BLACK, 0.75) << endl;
    out << "White self komi wins/draw=0.75: " << hist.currentSelfKomi(P_WHITE, 0.75) << endl;

    out << "Winner: " << PlayerIO::colorToChar(hist.winner) << endl;
    double score = hist.finalWhiteMinusBlackScore;
    out << "Final score: " << score << endl;

    double drawEquivsToTry[4] = {0.5, 0.3, 0.7, 1.0};
    for(int i = 0; i<4; i++) {
      double drawEquiv = drawEquivsToTry[i];
      string s = Global::strprintf("%.1f", drawEquiv);
      double scoreAdjusted = ScoreValue::whiteScoreDrawAdjust(score, drawEquiv, hist);
      double stdev = sqrt(std::max(0.0,ScoreValue::whiteScoreMeanSqOfScoreGridded(score,drawEquiv) - scoreAdjusted * scoreAdjusted));
      double sqrtBoardArea = board.sqrtBoardArea();
      double expectedScoreValue = ScoreValue::expectedWhiteScoreValue(scoreAdjusted, stdev, 0.0, 2.0, sqrtBoardArea);
      out << "WL Wins wins/draw=" << s << ": " << ScoreValue::whiteWinsOfWinner(hist.winner, drawEquiv) << endl;
      out << "Score wins/draw=" << s << ": " << scoreAdjusted << endl;
      out << "Score Stdev wins/draw=" << s << ": " << stdev << endl;
      out << "Score Util Smooth  wins/draw=" << s << ": " << ScoreValue::whiteScoreValueOfScoreSmooth(score, 0.0, 2.0, drawEquiv, sqrtBoardArea, hist) << endl;
      out << "Score Util SmootND wins/draw=" << s << ": " << ScoreValue::whiteScoreValueOfScoreSmoothNoDrawAdjust(score, 0.0, 2.0, sqrtBoardArea) << endl;
      out << "Score Util Gridded wins/draw=" << s << ": " << expectedScoreValue << endl;
      out << "Score Util GridInv wins/draw=" << s << ": " << ScoreValue::approxWhiteScoreOfScoreValueSmooth(expectedScoreValue,0.0,2.0, sqrtBoardArea) << endl;
    }
  };

  {
    const double power = 1.5;
    const double scale = 20.0;
    testAssert(ScoreValue::scoreMaximizingUtility(0.0,power,scale) == 0.0);
    testAssert(std::fabs(ScoreValue::scoreMaximizingUtility(20.0,power,scale) - (pow(2.0,1.5)-1.0)) < 1e-14);
    testAssert(std::fabs(ScoreValue::scoreMaximizingUtility(-60.0,power,scale) + 7.0) < 1e-14);
    testAssert(
      ScoreValue::scoreMaximizingUtility(83.5,power,scale) ==
      -ScoreValue::scoreMaximizingUtility(-83.5,power,scale)
    );

    Board board(19,19);
    Rules rules = Rules::getTrompTaylorish();
    BoardHistory hist(board,P_BLACK,rules,0);
    double lowerBound;
    double upperBound;
    ScoreValue::getScoreMaximizingUtilityLegalBounds(board,hist,lowerBound,upperBound);
    testAssert(lowerBound == -353.5);
    testAssert(upperBound == 368.5);

    Rules buttonRules = Rules::getTrompTaylorish();
    buttonRules.hasButton = true;
    buttonRules.komi = 7.0;
    BoardHistory buttonHist(board,P_BLACK,buttonRules,0);
    double buttonLowerBound;
    double buttonUpperBound;
    ScoreValue::getScoreMaximizingUtilityLegalBounds(
      board,buttonHist,buttonLowerBound,buttonUpperBound
    );
    testAssert(buttonLowerBound == -354.5);
    testAssert(buttonUpperBound == 368.5);

    BoardHistory bonusHist = hist;
    bonusHist.whiteBonusScore = 1.5;
    bonusHist.whiteHandicapBonusScore = -2.0;
    double bonusLowerBound;
    double bonusUpperBound;
    ScoreValue::getScoreMaximizingUtilityLegalBounds(
      board,bonusHist,bonusLowerBound,bonusUpperBound
    );
    testAssert(bonusLowerBound == -354.0);
    testAssert(bonusUpperBound == 368.0);

    double zeroStdevUtility = ScoreValue::expectedScoreMaximizingUtility(
      37.25,0.0,power,scale,lowerBound,upperBound
    );
    testAssert(std::fabs(zeroStdevUtility - ScoreValue::scoreMaximizingUtility(37.25,power,scale)) < 1e-14);
    testAssert(
      ScoreValue::expectedScoreMaximizingUtility(-1e100,0.0,power,scale,lowerBound,upperBound) ==
      ScoreValue::scoreMaximizingUtility(lowerBound,power,scale)
    );
    testAssert(
      ScoreValue::expectedScoreMaximizingUtility(1e100,0.0,power,scale,lowerBound,upperBound) ==
      ScoreValue::scoreMaximizingUtility(upperBound,power,scale)
    );

    double hugeTailUtility = ScoreValue::expectedScoreMaximizingUtility(
      0.0,1e12,power,scale,lowerBound,upperBound
    );
    double endpointAverage = 0.5 * (
      ScoreValue::scoreMaximizingUtility(lowerBound,power,scale) +
      ScoreValue::scoreMaximizingUtility(upperBound,power,scale)
    );
    testAssert(std::fabs(hugeTailUtility-endpointAverage) < 1e-6);
    double lowerTailProb;
    double upperTailProb;
    ScoreValue::getScoreMaximizingUtilityTailProbabilities(
      0.0,1e12,lowerBound,upperBound,lowerTailProb,upperTailProb
    );
    testAssert(std::fabs(lowerTailProb-0.5) < 1e-9);
    testAssert(std::fabs(upperTailProb-0.5) < 1e-9);

    double lossByOneUtility = -2.0 + ScoreValue::scoreMaximizingUtility(-1.0,power,scale);
    double lossByHundredUtility = -2.0 + ScoreValue::scoreMaximizingUtility(-100.0,power,scale);
    testAssert(lossByOneUtility > lossByHundredUtility);

    double utilityAtMean = ScoreValue::scoreMaximizingUtility(40.0,power,scale);
    double expectedRiskSeekingUtility = ScoreValue::expectedScoreMaximizingUtility(
      40.0,10.0,power,scale,lowerBound,upperBound
    );
    testAssert(expectedRiskSeekingUtility > utilityAtMean);
    double symmetricWhiteUtility = ScoreValue::expectedScoreMaximizingUtility(
      37.0,18.0,power,scale,-200.0,200.0
    );
    double symmetricBlackUtility = ScoreValue::expectedScoreMaximizingUtility(
      -37.0,18.0,power,scale,-200.0,200.0
    );
    testAssert(std::fabs(symmetricWhiteUtility+symmetricBlackUtility) < 1e-12);

    const double referenceCases[][2] = {
      {12.3,17.8},
      {40.0,10.0},
      {-120.0,85.0},
      {350.0,40.0},
      {500.0,700.0}
    };
    for(const auto& referenceCase: referenceCases) {
      double expected = ScoreValue::expectedScoreMaximizingUtility(
        referenceCase[0],referenceCase[1],power,scale,lowerBound,upperBound
      );
      double reference = slowExpectedScoreMaximizingUtility(
        referenceCase[0],referenceCase[1],power,scale,lowerBound,upperBound
      );
      testAssert(std::fabs(expected-reference) < 0.012);
    }

    const double endpointCases[][4] = {
      //scoreScale, mean, stdev, tolerance
      {5.0,0.0,10.0,0.003},
      {5.0,0.0,500.0,0.005},
      {1000.0,7.5,0.001,0.0001},
      {1000.0,350.0,40.0,0.0002}
    };
    for(const auto& endpointCase: endpointCases) {
      double endpointScale = endpointCase[0];
      double expected = ScoreValue::expectedScoreMaximizingUtility(
        endpointCase[1],endpointCase[2],1.0,endpointScale,lowerBound,upperBound
      );
      double reference = slowExpectedScoreMaximizingUtility(
        endpointCase[1],endpointCase[2],1.0,endpointScale,lowerBound,upperBound
      );
      if(std::fabs(expected-reference) >= endpointCase[3]) {
        cout << "Score utility endpoint regression: scale=" << endpointScale
             << " mean=" << endpointCase[1]
             << " stdev=" << endpointCase[2]
             << " expected=" << expected
             << " reference=" << reference
             << " tolerance=" << endpointCase[3] << endl;
      }
      testAssert(std::fabs(expected-reference) < endpointCase[3]);
    }

    bool rejectedUnsafePower = false;
    try {
      (void)ScoreValue::scoreMaximizingUtility(0.0,2.01,scale);
    }
    catch(const StringError&) {
      rejectedUnsafePower = true;
    }
    testAssert(rejectedUnsafePower);
    bool rejectedUnsafeScale = false;
    try {
      (void)ScoreValue::scoreMaximizingUtility(0.0,power,4.99);
    }
    catch(const StringError&) {
      rejectedUnsafeScale = true;
    }
    testAssert(rejectedUnsafeScale);

    Rules territoryRules = Rules::getSimpleTerritory();
    BoardHistory territoryHist(board,P_BLACK,territoryRules,0);
    bool threwForTerritory = false;
    try {
      ScoreValue::getScoreMaximizingUtilityLegalBounds(board,territoryHist,lowerBound,upperBound);
    }
    catch(const StringError& e) {
      threwForTerritory = std::string(e.what()).find("area scoring only") != std::string::npos;
    }
    testAssert(threwForTerritory);

    SearchParams params;
    testAssert(!params.useScoreMaximizingUtility);
    testAssert(params.scorePower == 1.5);
    testAssert(params.scoreScale == 20.0);
    testAssert(params.winWeight == 2.0);
    nlohmann::json paramsJson = params.changeableParametersToJson();
    testAssert(paramsJson["useScoreMaximizingUtility"] == false);
    testAssert(paramsJson["scorePower"] == 1.5);
    testAssert(paramsJson["scoreScale"] == 20.0);
    testAssert(paramsJson["winWeight"] == 2.0);
    SearchParams changedParams = params;
    changedParams.useScoreMaximizingUtility = true;
    testAssert(changedParams != params);
    testAssert(changedParams.getHash() != params.getHash());
    ostringstream paramsOut;
    streambuf* oldCoutBuf = cout.rdbuf(paramsOut.rdbuf());
    params.printParams(paramsOut);
    cout.rdbuf(oldCoutBuf);
    testAssert(paramsOut.str().find("useScoreMaximizingUtility: 0") != std::string::npos);
    testAssert(paramsOut.str().find("scorePower: 1.5") != std::string::npos);
    testAssert(paramsOut.str().find("scoreScale: 20") != std::string::npos);
    testAssert(paramsOut.str().find("winWeight: 2") != std::string::npos);

    map<string,string> configValues {
      {"numBots","2"},
      {"numSearchThreads","1"},
      {"useScoreMaximizingUtility","false"},
      {"scorePower","1.25"},
      {"scoreScale","25.0"},
      {"winWeight","2.5"},
      {"useScoreMaximizingUtility1","true"},
      {"scorePower1","2.0"},
      {"scoreScale1","10.0"},
      {"winWeight1","3.0"}
    };
    ConfigParser cfg(configValues);
    vector<SearchParams> loadedParams = Setup::loadParams(
      cfg,Setup::SETUP_FOR_OTHER,false,false
    );
    testAssert(loadedParams.size() == 2);
    testAssert(!loadedParams[0].useScoreMaximizingUtility);
    testAssert(loadedParams[0].scorePower == 1.25);
    testAssert(loadedParams[0].scoreScale == 25.0);
    testAssert(loadedParams[0].winWeight == 2.5);
    testAssert(loadedParams[1].useScoreMaximizingUtility);
    testAssert(loadedParams[1].scorePower == 2.0);
    testAssert(loadedParams[1].scoreScale == 10.0);
    testAssert(loadedParams[1].winWeight == 3.0);
  }

  {
    const char* name = "On-board even 9x9, komi 7.5";

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
ooooooooo
.........
.........
.........
xxxxxxxxx
.........
.........
)%%");

    Rules rules = Rules::getTrompTaylorish();
    BoardHistory hist(board,P_BLACK,rules,0);
    hist.endAndScoreGameNow(board);

    printScoreStats(board,hist);

    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
  }

  {
    const char* name = "On-board even 9x9, komi 7";

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
ooooooooo
.........
.........
.........
xxxxxxxxx
.........
.........
)%%");

    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 7.0;
    BoardHistory hist(board,P_BLACK,rules,0);
    hist.endAndScoreGameNow(board);

    printScoreStats(board,hist);

    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
  }

  {
    const char* name = "On-board black ahead 7 9x9, komi 7";

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
ooooooooo
.........
.........
xxxxxxx..
xxxxxxxxx
.........
.........
)%%");

    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 7.0;
    BoardHistory hist(board,P_BLACK,rules,0);
    hist.endAndScoreGameNow(board);

    printScoreStats(board,hist);

    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
  }


  {
    const char* name = "On-board even 5x5, komi 7";

    Board board = Board::parseBoard(5,5,R"%%(
.....
ooooo
.....
xxxxx
.....
)%%");

    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 7.0;
    BoardHistory hist(board,P_BLACK,rules,0);
    hist.endAndScoreGameNow(board);

    printScoreStats(board,hist);
    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
  }

  {
    const char* name = "Score value tables";
    cout << name << endl;
    for(int center = 0; center <= 5; center += 5) {
      for(int scale = 1; scale <= 2; scale++) {
        for(int b = 0; b<5; b++) {
          int xSizes[5] = {9, 13, 13, 13, 19};
          int ySizes[5] = {9, 9, 13, 19, 19};
          Board board(xSizes[b],ySizes[b]);
          cout << "center " << center << " scale " << scale << " x " << xSizes[b] << " y " << ySizes[b] << endl;
          for(int stdev = 0; stdev <= 5; stdev++) {
            for(double d = -8.0; d<=8.0; d += 0.5) {
              double scoreValue = ScoreValue::expectedWhiteScoreValue(d, stdev, center, scale, board.sqrtBoardArea());
              cout << Global::strprintf("%.3f ", scoreValue);
            }
            cout << endl;
          }
          cout << endl;
        }
        cout << endl;
      }
    }
  }

}
