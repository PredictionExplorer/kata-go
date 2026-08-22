#include "../core/global.h"
#include "../core/datetime.h"
#include "../core/fileutils.h"
#include "../core/hash.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"
#include "../dataio/loadmodel.h"
#include "../neuralnet/modelversion.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../program/selfplaymanager.h"
#include "../command/commandline.h"
#include "../core/test.h"
#include "../main.h"

#include <chrono>
#include <csignal>

using namespace std;

static std::atomic<bool> sigReceived(false);
static std::atomic<bool> shouldStop(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM) {
    sigReceived.store(true);
    shouldStop.store(true);
  }
}

//-----------------------------------------------------------------------------------------


int MainCmds::selfplay(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string modelsDir;
  string opponentModelsDir;
  string outputDir;
  int64_t maxGamesTotal = ((int64_t)1) << 62;
  try {
    KataGoCommandLine cmd("Generate training data via self play.");
    cmd.addConfigFileArg("","");
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> modelsDirArg("","models-dir","Dir to poll and load models from",true,string(),"DIR");
    TCLAP::ValueArg<string> opponentModelsDirArg(
      "","opponent-models-dir",
      "Optional frozen opponent model directory for focal extreme cohorts",
      false,string(),"DIR"
    );
    TCLAP::ValueArg<string> outputDirArg("","output-dir","Dir to output files",true,string(),"DIR");
    TCLAP::ValueArg<string> maxGamesTotalArg("","max-games-total","Terminate after this many games",false,string(),"NGAMES");
    cmd.add(modelsDirArg);
    cmd.add(opponentModelsDirArg);
    cmd.add(outputDirArg);
    cmd.add(maxGamesTotalArg);
    cmd.parseArgs(args);

    modelsDir = modelsDirArg.getValue();
    opponentModelsDir = opponentModelsDirArg.getValue();
    outputDir = outputDirArg.getValue();
    string maxGamesTotalStr = maxGamesTotalArg.getValue();
    if(maxGamesTotalStr != "") {
      bool suc = Global::tryStringToInt64(maxGamesTotalStr,maxGamesTotal);
      if(!suc || maxGamesTotal <= 0)
        throw StringError("-max-games-total must be a positive integer");
    }

    auto checkDirNonEmpty = [](const char* flag, const string& s) {
      if(s.length() <= 0)
        throw StringError("Empty directory specified for " + string(flag));
    };
    checkDirNonEmpty("models-dir",modelsDir);
    checkDirNonEmpty("output-dir",outputDir);

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  MakeDir::make(outputDir);
  MakeDir::make(modelsDir);

  Logger logger(&cfg);
  //Log to random file name to better support starting/stopping as well as multiple parallel runs
  logger.addFile(outputDir + "/log" + DateTime::getCompactDateTimeString() + "-" + Global::uint64ToHexString(seedRand.nextUInt64()) + ".log");

  logger.write("Self Play Engine starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

  //Load runner settings
  const int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Width and height of the board to use when writing data, typically 19
  const int dataBoardLen = cfg.getInt("dataBoardLen",3,Board::MAX_LEN);
  const int inputsVersion =
    cfg.contains("inputsVersion") ?
    cfg.getInt("inputsVersion",0,10000) :
    NNModelVersion::getInputsVersion(NNModelVersion::defaultModelVersion);
  //Max number of games that we will allow to be queued up and not written out
  const int maxDataQueueSize = cfg.getInt("maxDataQueueSize",1,1000000);
  const int maxRowsPerTrainFile = cfg.getInt("maxRowsPerTrainFile",1,100000000);
  const double firstFileRandMinProp = cfg.getDouble("firstFileRandMinProp",0.0,1.0);

  const int64_t logGamesEvery = cfg.getInt64("logGamesEvery",1,1000000);

  const bool switchNetsMidGame = cfg.getBool("switchNetsMidGame");
  const int extremeCohortSize =
    cfg.contains("extremeCohortSize") ?
    cfg.getInt("extremeCohortSize",0,64) :
    0;
  Player extremeCohortFocalPla = C_EMPTY;
  if(cfg.contains("extremeCohortFocalColor")) {
    if(!PlayerIO::tryParsePlayer(cfg.getString("extremeCohortFocalColor"),extremeCohortFocalPla))
      throw StringError("extremeCohortFocalColor must be black or white");
  }
  if(extremeCohortSize > 0 && extremeCohortFocalPla == C_EMPTY)
    throw StringError("extremeCohortFocalColor is required when extremeCohortSize is positive");
  if(extremeCohortSize > 0 && switchNetsMidGame)
    throw StringError("switchNetsMidGame must be false when extreme cohorts are enabled");
  SearchParams baseParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_OTHER);
  if(extremeCohortSize > 0) {
    if(opponentModelsDir.empty())
      throw StringError("-opponent-models-dir is required when extreme cohorts are enabled");
    if(!baseParams.useExpectedMaxScoreUtility || baseParams.useScoreMaximizingUtility)
      throw StringError("extreme cohorts require only useExpectedMaxScoreUtility");
    if(baseParams.extremeScoreGroupSize != extremeCohortSize)
      throw StringError("extremeCohortSize must equal extremeScoreGroupSize");
    if(baseParams.expectedMaxFocalPla != extremeCohortFocalPla)
      throw StringError("expectedMaxFocalColor must match extremeCohortFocalColor");
    if(maxDataQueueSize < extremeCohortSize || maxDataQueueSize % extremeCohortSize != 0) {
      throw StringError(
        "maxDataQueueSize must be a positive multiple of extremeCohortSize"
      );
    }
  }

  const string extremeCohortConfigIdentity =
    extremeCohortSize > 0 ?
    Global::uint64ToHexString(Hash::simpleHash(cfg.getAllKeyVals().c_str())) :
    "";
  const string extremeCohortRunIdentity =
    extremeCohortSize > 0 ?
    Global::uint64ToHexString(seedRand.nextUInt64()) :
    "";
  const ExtremeCohortSettings extremeCohortSettings(
    extremeCohortSize,
    extremeCohortFocalPla,
    extremeCohortConfigIdentity,
    extremeCohortRunIdentity
  );

  //Initialize object for randomizing game settings and running games
  const bool isDistributed = false;
  PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, isDistributed);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);
  bool autoCleanupAllButLatestIfUnused = true;
  SelfplayManager* manager = new SelfplayManager(
    maxDataQueueSize, &logger, logGamesEvery, autoCleanupAllButLatestIfUnused, extremeCohortSettings
  );

  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();

  //All neural evaluators, including the frozen stop-gradient opponent, require
  //the backend/session globals to exist before construction.
  Setup::initializeSession(cfg);

  NNEvaluator* frozenOpponentNNEval = NULL;
  string frozenOpponentModelName;
  string frozenOpponentModelFile;
  if(extremeCohortSettings.isEnabled()) {
    string frozenOpponentModelDir;
    time_t frozenOpponentModelTime;
    bool foundOpponent = LoadModel::findLatestModel(
      opponentModelsDir,
      logger,
      frozenOpponentModelName,
      frozenOpponentModelFile,
      frozenOpponentModelDir,
      frozenOpponentModelTime
    );
    if(!foundOpponent || frozenOpponentModelFile == "/dev/null")
      throw StringError("Could not load a frozen opponent model for extreme cohorts");

    const int expectedConcurrentEvals = cfg.getInt("numSearchThreads") * numGameThreads;
    const bool defaultRequireExactNNLen =
      minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
    const int defaultMaxBatchSize = -1;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    Rand opponentRand;
    frozenOpponentNNEval = Setup::initializeNNEvaluator(
      frozenOpponentModelName,frozenOpponentModelFile,expectedSha256,cfg,logger,opponentRand,expectedConcurrentEvals,
      maxBoardXSizeUsed,maxBoardYSizeUsed,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_OTHER
    );
    logger.write(
      "Loaded frozen extreme-cohort opponent " + frozenOpponentModelName +
      " from: " + frozenOpponentModelFile
    );
  }

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting self play");
  if(extremeCohortSettings.isEnabled()) {
    logger.write(
      "Extreme cohorts enabled: N=" + Global::intToString(extremeCohortSettings.groupSize) +
      " focal=" + PlayerIO::playerToString(extremeCohortSettings.focalPla) +
      " config=" + extremeCohortSettings.configIdentity +
      " run=" + extremeCohortSettings.runIdentity
    );
  }
  if(!logger.isLoggingToStdout())
    cout << "Loaded all config stuff, starting self play" << endl;

  //Time the whole self-play run for reporting overall computational throughput.
  ClockTimer selfplayTimer;

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);


  //Returns true if a new net was loaded.
  auto loadLatestNeuralNetIntoManager =
    [inputsVersion,&manager,maxRowsPerTrainFile,firstFileRandMinProp,dataBoardLen,
     &modelsDir,&outputDir,&logger,&cfg,numGameThreads,
     &extremeCohortSettings,
     &frozenOpponentModelName,&frozenOpponentModelFile,
     minBoardXSizeUsed,maxBoardXSizeUsed,minBoardYSizeUsed,maxBoardYSizeUsed](const string* lastNetName) -> bool {

    string modelName;
    string modelFile;
    string modelDir;
    time_t modelTime;
    bool foundModel = LoadModel::findLatestModel(modelsDir, logger, modelName, modelFile, modelDir, modelTime);

    //No new neural nets yet
    if(!foundModel || (lastNetName != NULL && *lastNetName == modelName))
      return false;
    if(modelName == "random" && lastNetName != NULL && *lastNetName != "random") {
      logger.write("WARNING: " + *lastNetName + " was the previous model, but now no model was found. Continuing with prev model instead of using random");
      return false;
    }

    logger.write("Found new neural net " + modelName);

    const int expectedConcurrentEvals = cfg.getInt("numSearchThreads") * numGameThreads;
    const bool defaultRequireExactNNLen = minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
    const int defaultMaxBatchSize = -1;
    const bool disableFP16 = false;
    const string expectedSha256 = "";

    Rand rand;
     NNEvaluator* nnEval = Setup::initializeNNEvaluator(
      modelName,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
      maxBoardXSizeUsed,maxBoardYSizeUsed,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_OTHER
    );
    logger.write("Loaded latest neural net " + modelName + " from: " + modelFile);

    string modelOutputDir = outputDir + "/" + modelName;
    string sgfOutputDir = modelOutputDir + "/sgfs";
    string tdataOutputDir = modelOutputDir + "/tdata";

    //Try repeatedly to make directories, in case the filesystem is unhappy with us as we try to make the same dirs as another process.
    //Wait a random amount of time in between each failure.
    int maxTries = 5;
    for(int i = 0; i<maxTries; i++) {
      bool success = false;
      try {
        MakeDir::make(modelOutputDir);
        MakeDir::make(sgfOutputDir);
        MakeDir::make(tdataOutputDir);
        success = true;
      }
      catch(const StringError& e) {
        logger.write(string("WARNING, error making directories, trying again shortly: ") + e.what());
        success = false;
      }

      if(success)
        break;
      else {
        if(i == maxTries-1) {
          logger.write("ERROR: Could not make selfplay model directories, is something wrong with the filesystem?");
          //Just give up and wait for the next model.
          return false;
        }
        double sleepTime = 10.0 + rand.nextDouble() * 30.0;
        std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
        continue;
      }
    }

    const string selfplayConfigFile =
      modelOutputDir + "/selfplay-" + Global::uint64ToHexString(rand.nextUInt64()) + ".cfg";
    {
      ofstream out;
      FileUtils::open(out,selfplayConfigFile);
      out << cfg.getContents();
      out.close();
    }
    if(extremeCohortSettings.isEnabled()) {
      ofstream out;
      FileUtils::open(
        out,
        modelOutputDir + "/extreme-cohort-" + extremeCohortSettings.runIdentity + ".manifest.cfg"
      );
      out << "schemaVersion = " << ExtremeCohortData::METADATA_VERSION << "\n";
      out << "mode = focal-extreme-leave-one-out\n";
      out << "runIdentity = " << extremeCohortSettings.runIdentity << "\n";
      out << "configIdentity = " << extremeCohortSettings.configIdentity << "\n";
      out << "cohortGroupSize = " << extremeCohortSettings.groupSize << "\n";
      out << "focalColor = " << PlayerIO::playerToStringShort(extremeCohortSettings.focalPla) << "\n";
      out << "focalModelIdentity = " << modelName << "\n";
      out << "opponentModelIdentity = " << frozenOpponentModelName << "\n";
      out << "modelFile = " << modelFile << "\n";
      out << "opponentModelFile = " << frozenOpponentModelFile << "\n";
      out << "selfplayConfigFile = " << selfplayConfigFile << "\n";
      out << "cohortIdScope = runIdentity\n";
      out << "cohortIdStart = " << extremeCohortSettings.cohortIdStart() << "\n";
      out << "assignmentOrder = launch-order-per-model-generation\n";
      out << "scorePerspective = focal-color-final-margin\n";
      out << "credit = N1:1;Ngt1:max(0,S_i-max_other_S)\n";
      out << "singletonMaxOther = 0\n";
      out << "incompleteCohortPolicy = drop\n";
      out << "globalTargetsChannels = 68-79\n";
      out.close();
    }

    //Note that this inputsVersion passed here is NOT necessarily the same as the one used in the neural net self play, it
    //simply controls the input feature version for the written data
    TrainingDataWriter* tdataWriter = new TrainingDataWriter(
      tdataOutputDir, inputsVersion, maxRowsPerTrainFile, firstFileRandMinProp, dataBoardLen, dataBoardLen, Global::uint64ToHexString(rand.nextUInt64()));
    ofstream* sgfOut = NULL;
    if(sgfOutputDir.length() > 0) {
      sgfOut = new ofstream();
      FileUtils::open(*sgfOut, sgfOutputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".sgfs");
    }

    logger.write("Model loading loop thread loaded new neural net " + nnEval->getModelName());
    manager->loadModelAndStartDataWriting(nnEval, tdataWriter, sgfOut);
    return true;
  };

  //Initialize the initial neural net
  {
    bool success = loadLatestNeuralNetIntoManager(NULL);
    if(!success)
      throw StringError("Either could not load latest neural net or access/write appopriate directories");
  }

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  //Shared across all game loop threads
  std::atomic<int64_t> numGamesStarted(0);
  ForkData* forkData = new ForkData();
  auto gameLoop = [
    &gameRunner,
    &manager,
    &logger,
    switchNetsMidGame,
    &numGamesStarted,
    &forkData,
    maxGamesTotal,
    &baseParams,
    &gameSeedBase,
    &extremeCohortSettings,
    &frozenOpponentNNEval,
    &frozenOpponentModelName
  ](int threadIdx) {
    auto shouldStopFunc = []() noexcept {
      return shouldStop.load();
    };
    WaitableFlag* shouldPause = nullptr;

    string prevModelName;
    Rand thisLoopSeedRand;
    while(true) {
      if(shouldStop.load())
        break;
      NNEvaluator* nnEval = manager->acquireLatest();
      testAssert(nnEval != NULL);

      if(prevModelName != nnEval->getModelName()) {
        prevModelName = nnEval->getModelName();
        logger.write("Game loop thread " + Global::intToString(threadIdx) + " starting game on new neural net: " + prevModelName);
      }

      //Callback that runGame will call periodically to ask us if we have a new neural net
      std::function<NNEvaluator*()> checkForNewNNEval = [&manager,&nnEval,&prevModelName,&logger,&threadIdx]() -> NNEvaluator* {
        NNEvaluator* newNNEval = manager->acquireLatest();
        testAssert(newNNEval != NULL);
        if(newNNEval == nnEval) {
          manager->release(newNNEval);
          return NULL;
        }
        manager->release(nnEval);

        nnEval = newNNEval;
        prevModelName = nnEval->getModelName();
        logger.write("Game loop thread " + Global::intToString(threadIdx) + " changing midgame to new neural net: " + prevModelName);
        return nnEval;
      };

      FinishedGameData* gameData = NULL;
      bool hasDataWriteReservation = false;
      if(extremeCohortSettings.isEnabled()) {
        hasDataWriteReservation =
          manager->waitForDataToWriteCapacity(nnEval,shouldStopFunc);
        if(!hasDataWriteReservation) {
          manager->release(nnEval);
          break;
        }
      }

      int64_t gameIdx = numGamesStarted.fetch_add(1,std::memory_order_acq_rel);
      if(gameIdx < maxGamesTotal) {
        MatchPairer::BotSpec focalBotSpec;
        focalBotSpec.botIdx = 0;
        focalBotSpec.botName = nnEval->getModelName();
        focalBotSpec.nnEval = nnEval;
        focalBotSpec.baseParams = baseParams;
        MatchPairer::BotSpec botSpecB = focalBotSpec;
        MatchPairer::BotSpec botSpecW = focalBotSpec;

        if(extremeCohortSettings.isEnabled()) {
          testAssert(frozenOpponentNNEval != NULL);
          MatchPairer::BotSpec opponentBotSpec;
          opponentBotSpec.botIdx = 1;
          opponentBotSpec.botName = frozenOpponentModelName;
          opponentBotSpec.nnEval = frozenOpponentNNEval;
          opponentBotSpec.baseParams = baseParams;
          if(extremeCohortSettings.focalPla == P_BLACK)
            botSpecW = opponentBotSpec;
          else
            botSpecB = opponentBotSpec;
        }

        ExtremeCohortData extremeCohortAssignment;
        if(extremeCohortSettings.isEnabled()) {
          extremeCohortAssignment =
            manager->countOneGameStartedAndGetCohort(nnEval,frozenOpponentModelName);
        }
        else {
          manager->countOneGameStarted(nnEval);
        }

        string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        gameData = gameRunner->runGame(
          seed, botSpecB, botSpecW, forkData, NULL, logger,
          shouldStopFunc,
          shouldPause,
          (switchNetsMidGame ? checkForNewNNEval : nullptr),
          nullptr,
          nullptr
        );
        if(gameData != NULL && extremeCohortSettings.isEnabled())
          gameData->extremeCohort = extremeCohortAssignment;
      }

      //NULL gamedata will happen when the game is interrupted by shouldStop, which means we should also stop.
      //Or when we run out of total games.
      bool shouldContinue = gameData != NULL;
      //Note that if we've gotten a newNNEval, we're actually pushing the game as data for the new one, rather than the old one!
      if(gameData != NULL) {
        manager->enqueueDataToWrite(nnEval,gameData);
        hasDataWriteReservation = false;
      }
      if(hasDataWriteReservation)
        manager->cancelDataToWriteReservation(nnEval);

      manager->release(nnEval);

      if(!shouldContinue)
        break;
    }

    logger.write("Game loop thread " + Global::intToString(threadIdx) + " terminating");
  };
  auto gameLoopProtected = [&logger,&gameLoop](int threadIdx) {
    Logger::logThreadUncaught("game loop", &logger, [&](){ gameLoop(threadIdx); });
  };

  //Looping thread for polling for new neural nets and loading them in
  std::mutex modelLoadMutex;
  std::condition_variable modelLoadSleepVar;
  auto modelLoadLoop = [&modelLoadMutex,&modelLoadSleepVar,&logger,&manager,&loadLatestNeuralNetIntoManager]() {
    logger.write("Model loading loop thread starting");

    while(true) {
      if(shouldStop.load())
        break;
      string lastNetName = manager->getLatestModelName();
      bool success = loadLatestNeuralNetIntoManager(&lastNetName);
      (void)success;

      if(shouldStop.load())
        break;

      //Sleep for a while and then re-poll
      std::unique_lock<std::mutex> lock(modelLoadMutex);
      modelLoadSleepVar.wait_for(lock, std::chrono::seconds(20), [](){return shouldStop.load();});
    }

    logger.write("Model loading loop thread terminating");
  };
  auto modelLoadLoopProtected = [&logger,&modelLoadLoop]() {
    Logger::logThreadUncaught("model load loop", &logger, modelLoadLoop);
  };

  vector<std::thread> threads;
  threads.reserve(numGameThreads);
  for(int i = 0; i<numGameThreads; i++) {
    threads.emplace_back(gameLoopProtected,i);
  }
  std::thread modelLoadLoopThread(modelLoadLoopProtected);

  //Wait for all game threads to stop
  for(int i = 0; i<threads.size(); i++)
    threads[i].join();

  //If by now somehow shouldStop is not true, set it to be true since all game threads are toast
  shouldStop.store(true);

  //Wake up the model loading thread rather than waiting for it to wake up on its own, and
  //wait for it to die.
  {
    //Lock so that we don't race where we notify the loading thread to wake when it's still in
    //its own critical section but not yet slept, and to ensure the two agree on shouldStop.
    std::lock_guard<std::mutex> lock(modelLoadMutex);
    modelLoadSleepVar.notify_all();
  }
  modelLoadLoopThread.join();

  //At this point, nothing else except possibly data write loops are running, within the selfplay manager.
  delete manager;
  if(frozenOpponentNNEval != NULL)
    delete frozenOpponentNNEval;

  //Overall self-play totals (per-model NN/data/moves breakdowns are logged above by the manager).
  logger.write("Total games: " + Global::int64ToString(numGamesStarted.load(std::memory_order_relaxed)));
  logger.write("Total selfplay runtime (seconds): " + Global::doubleToString(selfplayTimer.getSeconds()));

  //Delete and clean up everything else
  NeuralNet::globalCleanup();
  delete forkData;
  delete gameRunner;
  ScoreValue::freeTables();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}
