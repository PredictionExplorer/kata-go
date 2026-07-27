#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../search/asyncbot.h"
#include "../search/patternbonustable.h"
#include "../search/reportedsearchvalues.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../core/test.h"
#include "../main.h"

#include <csignal>

using namespace std;

struct DeterministicMatchScheduleEntry {
  string scheduleId;
  string gameId;
  string pairId;
  string positionId;
  string seed;
  int blackBotIdx;
  int whiteBotIdx;
  Sgf::PositionSample startPosition;
};

static vector<DeterministicMatchScheduleEntry> loadDeterministicMatchSchedule(const string& fileName) {
  vector<string> lines = FileUtils::readFileLines(fileName,'\n');
  vector<DeterministicMatchScheduleEntry> schedule;
  set<string> gameIds;
  string commonScheduleId;

  for(size_t i = 0; i<lines.size(); i++) {
    string line = Global::trim(lines[i]);
    if(line.empty())
      continue;

    DeterministicMatchScheduleEntry entry;
    try {
      nlohmann::json data = nlohmann::json::parse(line);
      if(!data.is_object())
        throw StringError("schedule row is not a JSON object");
      if(data.at("schemaVersion").get<int>() != 1)
        throw StringError("unsupported schemaVersion (expected 1)");

      entry.scheduleId = data.at("scheduleId").get<string>();
      entry.gameId = data.at("gameId").get<string>();
      entry.seed = data.at("seed").get<string>();
      entry.blackBotIdx = data.at("blackBot").get<int>();
      entry.whiteBotIdx = data.at("whiteBot").get<int>();
      if(data.find("pairId") != data.end())
        entry.pairId = data.at("pairId").get<string>();
      if(data.find("positionId") != data.end())
        entry.positionId = data.at("positionId").get<string>();

      const nlohmann::json& startPositionJson = data.at("startPosition");
      if(!startPositionJson.is_object())
        throw StringError("startPosition is not a JSON object");
      entry.startPosition = Sgf::PositionSample::ofJsonLine(startPositionJson.dump());
    }
    catch(const nlohmann::detail::exception& e) {
      throw StringError(
        "Error parsing deterministic schedule " + fileName + " line " +
        Global::uint64ToString(i+1) + ": " + e.what()
      );
    }
    catch(const StringError& e) {
      throw StringError(
        "Error parsing deterministic schedule " + fileName + " line " +
        Global::uint64ToString(i+1) + ": " + e.what()
      );
    }

    if(entry.scheduleId.empty())
      throw StringError("Empty scheduleId in deterministic schedule " + fileName + " line " + Global::uint64ToString(i+1));
    if(entry.gameId.empty())
      throw StringError("Empty gameId in deterministic schedule " + fileName + " line " + Global::uint64ToString(i+1));
    if(entry.seed.empty())
      throw StringError("Empty seed in deterministic schedule " + fileName + " line " + Global::uint64ToString(i+1));
    if(commonScheduleId.empty())
      commonScheduleId = entry.scheduleId;
    else if(entry.scheduleId != commonScheduleId)
      throw StringError("All rows in deterministic schedule must have the same scheduleId");
    if(contains(gameIds,entry.gameId))
      throw StringError("Duplicate gameId in deterministic schedule: " + entry.gameId);
    gameIds.insert(entry.gameId);
    schedule.push_back(entry);
  }

  if(schedule.empty())
    throw StringError("Deterministic schedule is empty: " + fileName);
  return schedule;
}

static nlohmann::json makeMatchResultJson(
  const FinishedGameData& gameData,
  const string& seed,
  const DeterministicMatchScheduleEntry* scheduleEntry
) {
  const BoardHistory& hist = gameData.endHist;
  nlohmann::json result;
  result["schemaVersion"] = 1;
  if(scheduleEntry != NULL) {
    result["scheduleId"] = scheduleEntry->scheduleId;
    result["gameId"] = scheduleEntry->gameId;
    if(!scheduleEntry->pairId.empty())
      result["pairId"] = scheduleEntry->pairId;
    if(!scheduleEntry->positionId.empty())
      result["positionId"] = scheduleEntry->positionId;
  }
  else {
    result["scheduleId"] = nullptr;
    result["gameId"] = seed;
  }
  result["seed"] = seed;
  result["blackBot"] = gameData.bName;
  result["whiteBot"] = gameData.wName;
  result["blackBotIndex"] = gameData.bIdx;
  result["whiteBotIndex"] = gameData.wIdx;

  nlohmann::json board;
  board["xSize"] = gameData.startBoard.x_size;
  board["ySize"] = gameData.startBoard.y_size;
  result["board"] = board;
  result["rules"] = hist.rules.toJson();
  result["komi"] = hist.rules.komi;

  result["finalResult"] = gameData.hitTurnLimit ? "turn_limit" : WriteSgf::gameResultNoSgfTag(hist);
  if(hist.isScored)
    result["finalWhiteMinusBlackScore"] = hist.finalWhiteMinusBlackScore;
  else
    result["finalWhiteMinusBlackScore"] = nullptr;

  if(gameData.hitTurnLimit || hist.isNoResult)
    result["winner"] = nullptr;
  else if(hist.winner == P_BLACK)
    result["winner"] = "B";
  else if(hist.winner == P_WHITE)
    result["winner"] = "W";
  else
    result["winner"] = "draw";

  result["moveCount"] = gameData.bMoveCount + gameData.wMoveCount;
  result["blackMoveCount"] = gameData.bMoveCount;
  result["whiteMoveCount"] = gameData.wMoveCount;
  result["startTurnNumber"] = gameData.startHist.initialTurnNumber + (int64_t)gameData.startHist.moveHistory.size();
  result["hitTurnLimit"] = gameData.hitTurnLimit;
  result["resignation"] = hist.isResignation;
  result["noResult"] = hist.isNoResult;
  result["scored"] = hist.isScored;
  result["gameHash"] = gameData.gameHash.toString();
  return result;
}


static std::atomic<bool> sigReceived(false);
static std::atomic<bool> shouldStop(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM) {
    sigReceived.store(true);
    shouldStop.store(true);
  }
}

int MainCmds::match(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string logFile;
  string sgfOutputDir;
  string matchResultJsonlFile;
  string matchMoveJsonlFile;
  try {
    KataGoCommandLine cmd("Play different nets against each other with different search settings in a match or tournament.");
    cmd.addConfigFileArg("","match_example.cfg");

    TCLAP::ValueArg<string> logFileArg("","log-file","Log file to output to",false,string(),"FILE");
    TCLAP::ValueArg<string> sgfOutputDirArg("","sgf-output-dir","Dir to output sgf files",false,string(),"DIR");

    cmd.add(logFileArg);
    cmd.add(sgfOutputDirArg);

    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    logFile = logFileArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();

    cmd.getConfig(cfg);
    if(cfg.contains("matchResultJsonlFile"))
      matchResultJsonlFile = cfg.getString("matchResultJsonlFile");
    if(cfg.contains("matchMoveJsonlFile"))
      matchMoveJsonlFile = cfg.getString("matchMoveJsonlFile");
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger(&cfg);
  logger.addFile(logFile);

  logger.write("Match Engine starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

  //Load per-bot search config, first, which also tells us how many bots we're running
  vector<SearchParams> paramss = Setup::loadParams(cfg,Setup::SETUP_FOR_MATCH);
  assert(paramss.size() > 0);
  int numBots = (int)paramss.size();

  //Figure out all pairs of bots that will be playing.
  std::vector<std::pair<int,int>> matchupsPerRound;
  {
    //Load a filter on what bots we actually want to run. By default, include everything.
    vector<bool> includeBot(numBots);
    if(cfg.contains("includeBots")) {
      vector<int> includeBotIdxs = cfg.getInts("includeBots",0,Setup::MAX_BOT_PARAMS_FROM_CFG);
      for(int i = 0; i<numBots; i++) {
        if(contains(includeBotIdxs,i))
          includeBot[i] = true;
      }
    }
    else {
      for(int i = 0; i<numBots; i++) {
        includeBot[i] = true;
      }
    }

    std::vector<int> secondaryBotIdxs;
    if(cfg.contains("secondaryBots"))
      secondaryBotIdxs = cfg.getInts("secondaryBots",0,Setup::MAX_BOT_PARAMS_FROM_CFG);
    for(int i = 0; i<secondaryBotIdxs.size(); i++)
      if(secondaryBotIdxs[i] < 0 || secondaryBotIdxs[i] >= numBots)
        throw StringError("secondaryBots value " + Global::intToString(secondaryBotIdxs[i]) + " is out of range, numBots is " + Global::intToString(numBots));

    for(int i = 0; i<numBots; i++) {
      if(!includeBot[i])
        continue;
      for(int j = 0; j<numBots; j++) {
        if(!includeBot[j])
          continue;
        if(i < j && !(contains(secondaryBotIdxs,i) && contains(secondaryBotIdxs,j))) {
          matchupsPerRound.emplace_back(i,j);
          matchupsPerRound.emplace_back(j,i);
        }
      }
    }

    if(cfg.contains("extraPairs")) {
      std::vector<std::pair<int,int>> pairs = cfg.getNonNegativeIntDashedPairs("extraPairs",0,numBots-1);
      for(const std::pair<int,int>& pair: pairs) {
        int p0 = pair.first;
        int p1 = pair.second;
        if(cfg.contains("extraPairsAreOneSidedBW") && cfg.getBool("extraPairsAreOneSidedBW")) {
          matchupsPerRound.emplace_back(p0,p1);
        }
        else {
          matchupsPerRound.emplace_back(p0,p1);
          matchupsPerRound.emplace_back(p1,p0);
        }
      }
    }
  }

  //Load the names of the bots and which model each bot is using
  vector<string> nnModelFilesByBot(numBots);
  vector<string> botNames(numBots);
  for(int i = 0; i<numBots; i++) {
    string idxStr = Global::intToString(i);

    if(cfg.contains("botName"+idxStr))
      botNames[i] = cfg.getString("botName"+idxStr);
    else if(numBots == 1)
      botNames[i] = cfg.getString("botName");
    else
      throw StringError("If more than one bot, must specify botName0, botName1,... individually");

    if(cfg.contains("nnModelFile"+idxStr))
      nnModelFilesByBot[i] = cfg.getString("nnModelFile"+idxStr);
    else
      nnModelFilesByBot[i] = cfg.getString("nnModelFile");
  }

  string deterministicScheduleFile;
  vector<DeterministicMatchScheduleEntry> deterministicSchedule;
  if(cfg.contains("deterministicScheduleFile")) {
    deterministicScheduleFile = cfg.getString("deterministicScheduleFile");
    if(deterministicScheduleFile.empty())
      throw StringError("deterministicScheduleFile must not be empty");
    deterministicSchedule = loadDeterministicMatchSchedule(deterministicScheduleFile);
    logger.write(
      "Loaded deterministic schedule " + deterministicSchedule[0].scheduleId + " with " +
      Global::uint64ToString(deterministicSchedule.size()) + " games"
    );
  }
  const bool deterministicScheduleEnabled = !deterministicSchedule.empty();

  vector<bool> botIsUsed(numBots);
  if(deterministicScheduleEnabled) {
    for(const DeterministicMatchScheduleEntry& entry : deterministicSchedule) {
      if(entry.blackBotIdx < 0 || entry.blackBotIdx >= numBots)
        throw StringError("blackBot is out of range in deterministic game " + entry.gameId);
      if(entry.whiteBotIdx < 0 || entry.whiteBotIdx >= numBots)
        throw StringError("whiteBot is out of range in deterministic game " + entry.gameId);
      botIsUsed[entry.blackBotIdx] = true;
      botIsUsed[entry.whiteBotIdx] = true;
    }
  }
  else {
    for(const std::pair<int,int>& pair : matchupsPerRound) {
      botIsUsed[pair.first] = true;
      botIsUsed[pair.second] = true;
    }
  }

  //Dedup and load each necessary model exactly once
  vector<string> nnModelFiles;
  vector<int> whichNNModel(numBots);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;

    const string& desiredFile = nnModelFilesByBot[i];
    int alreadyFoundIdx = -1;
    for(int j = 0; j<nnModelFiles.size(); j++) {
      if(nnModelFiles[j] == desiredFile) {
        alreadyFoundIdx = j;
        break;
      }
    }
    if(alreadyFoundIdx != -1)
      whichNNModel[i] = alreadyFoundIdx;
    else {
      whichNNModel[i] = (int)nnModelFiles.size();
      nnModelFiles.push_back(desiredFile);
    }
  }

  //Load match runner settings
  int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());
  if(deterministicScheduleEnabled) {
    if(numGameThreads != 1)
      throw StringError("deterministicScheduleFile requires numGameThreads = 1");
    for(int i = 0; i<numBots; i++) {
      if(!botIsUsed[i])
        continue;
      if(paramss[i].numThreads != 1)
        throw StringError("deterministicScheduleFile requires numSearchThreads = 1 for every scheduled bot");
      if(paramss[i].maxTime < 1.0e19)
        throw StringError("deterministicScheduleFile requires visit/playout limits, not maxTime");
    }
  }

  //Work out an upper bound on how many concurrent nneval requests we could end up making.
  int expectedConcurrentEvals;
  {
    //Work out the max threads any one bot uses
    int maxBotThreads = 0;
    for(int i = 0; i<numBots; i++)
      if(paramss[i].numThreads > maxBotThreads)
        maxBotThreads = paramss[i].numThreads;
    //Mutiply by the number of concurrent games we could have
    expectedConcurrentEvals = maxBotThreads * numGameThreads;
  }

  //Initialize object for randomizing game settings and running games
  PlaySettings playSettings = PlaySettings::loadForMatch(cfg);
  if(deterministicScheduleEnabled && playSettings.initGamesWithPolicy)
    throw StringError("deterministicScheduleFile requires initGamesWithPolicy = false");
  GameRunner* gameRunner =
    deterministicScheduleEnabled ?
    new GameRunner(cfg, "deterministic-match-schedule:" + deterministicSchedule[0].scheduleId, playSettings, logger) :
    new GameRunner(cfg, playSettings, logger);
  if(deterministicScheduleEnabled) {
    for(const DeterministicMatchScheduleEntry& entry : deterministicSchedule) {
      if(!gameRunner->getGameInitializer()->isAllowedBSize(entry.startPosition.board.x_size,entry.startPosition.board.y_size)) {
        throw StringError(
          "Board size in deterministic game " + entry.gameId + " is not included in bSizes or bSizesXY"
        );
      }
    }
  }
  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();

  //Initialize neural net inference engine globals, and load models
  Setup::initializeSession(cfg);
  const vector<string>& nnModelNames = nnModelFiles;
  const int defaultMaxBatchSize = -1;
  const bool defaultRequireExactNNLen = minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
  const bool disableFP16 = false;
  const vector<string> expectedSha256s;
  vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators(
    nnModelNames,nnModelFiles,expectedSha256s,cfg,logger,seedRand,expectedConcurrentEvals,
    maxBoardXSizeUsed,maxBoardYSizeUsed,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
    Setup::SETUP_FOR_MATCH
  );
  logger.write("Loaded neural net");

  vector<NNEvaluator*> nnEvalsByBot(numBots);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;
    nnEvalsByBot[i] = nnEvals[whichNNModel[i]];
  }

  std::vector<std::unique_ptr<PatternBonusTable>> patternBonusTables = Setup::loadAvoidSgfPatternBonusTables(cfg,logger);
  testAssert(patternBonusTables.size() == numBots);

  //Initialize object for randomly pairing bots
  int64_t numGamesTotal;
  int64_t deterministicLogGamesEvery = 0;
  MatchPairer* matchPairer = NULL;
  if(deterministicScheduleEnabled) {
    numGamesTotal = (int64_t)deterministicSchedule.size();
    deterministicLogGamesEvery = cfg.getInt64("logGamesEvery",1,1000000);
    if(cfg.contains("numGamesTotal")) {
      int64_t configuredNumGames = cfg.getInt64("numGamesTotal",1,((int64_t)1) << 62);
      if(configuredNumGames != numGamesTotal)
        throw StringError("numGamesTotal must equal the number of deterministic schedule rows");
    }
  }
  else {
    numGamesTotal = cfg.getInt64("numGamesTotal",1,((int64_t)1) << 62);
    matchPairer = new MatchPairer(cfg,numBots,botNames,nnEvalsByBot,paramss,matchupsPerRound,numGamesTotal);
  }

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;
    Setup::maybeWarnHumanSLParams(paramss[i],nnEvalsByBot[i],NULL,cerr,&logger);
  }

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting matches");
  if(!logger.isLoggingToStdout())
    cout << "Loaded all config stuff, starting matches" << endl;

  if(sgfOutputDir != string())
    MakeDir::make(sgfOutputDir);

  ofstream* matchResultOut = NULL;
  if(!matchResultJsonlFile.empty()) {
    matchResultOut = new ofstream();
    FileUtils::open(*matchResultOut,matchResultJsonlFile);
  }
  ofstream* matchMoveOut = NULL;
  if(!matchMoveJsonlFile.empty()) {
    matchMoveOut = new ofstream();
    FileUtils::open(*matchMoveOut,matchMoveJsonlFile);
  }

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);


  std::mutex statsMutex;
  std::mutex resultMutex;
  std::mutex scheduleMutex;
  size_t nextScheduleIdx = 0;
  int64_t gameCount = 0;
  std::map<string,double> timeUsedByBotMap;
  std::map<string,double> movesByBotMap;

  auto runMatchLoop = [
    &gameRunner,&matchPairer,&sgfOutputDir,&logger,&gameSeedBase,&patternBonusTables,
    &deterministicSchedule,&deterministicScheduleEnabled,&deterministicLogGamesEvery,
    &nextScheduleIdx,&scheduleMutex,
    &botNames,&nnEvalsByBot,&paramss,&matchResultOut,&matchMoveOut,&resultMutex,
    &statsMutex,&gameCount,&timeUsedByBotMap,&movesByBotMap
  ](
    uint64_t threadHash
  ) {
    ofstream* sgfOut = NULL;
    if(sgfOutputDir.length() > 0) {
      sgfOut = new ofstream();
      string sgfFileName =
        deterministicScheduleEnabled ? "games.sgfs" : Global::uint64ToHexString(threadHash) + ".sgfs";
      FileUtils::open(*sgfOut, sgfOutputDir + "/" + sgfFileName);
    }
    auto shouldStopFunc = []() noexcept {
      return shouldStop.load();
    };
    WaitableFlag* shouldPause = nullptr;

    Rand thisLoopSeedRand;
    while(true) {
      if(shouldStop.load())
        break;

      FinishedGameData* gameData = NULL;
      const DeterministicMatchScheduleEntry* scheduleEntry = NULL;
      string seed;

      MatchPairer::BotSpec botSpecB;
      MatchPairer::BotSpec botSpecW;
      bool hasMatchup = false;
      if(deterministicScheduleEnabled) {
        {
          std::lock_guard<std::mutex> lock(scheduleMutex);
          if(nextScheduleIdx < deterministicSchedule.size()) {
            scheduleEntry = &deterministicSchedule[nextScheduleIdx];
            nextScheduleIdx += 1;
            if(nextScheduleIdx % deterministicLogGamesEvery == 0)
              logger.write("Started " + Global::uint64ToString(nextScheduleIdx) + " games");
          }
        }
        if(scheduleEntry != NULL) {
          botSpecB.botIdx = scheduleEntry->blackBotIdx;
          botSpecB.botName = botNames[botSpecB.botIdx];
          botSpecB.nnEval = nnEvalsByBot[botSpecB.botIdx];
          botSpecB.baseParams = paramss[botSpecB.botIdx];
          botSpecW.botIdx = scheduleEntry->whiteBotIdx;
          botSpecW.botName = botNames[botSpecW.botIdx];
          botSpecW.nnEval = nnEvalsByBot[botSpecW.botIdx];
          botSpecW.baseParams = paramss[botSpecW.botIdx];
          seed = scheduleEntry->seed;
          hasMatchup = true;
        }
      }
      else if(matchPairer->getMatchup(botSpecB, botSpecW, logger)) {
        seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        hasMatchup = true;
      }

      if(hasMatchup) {
        std::function<void(const MatchPairer::BotSpec&, Search*)> afterInitialization = [&patternBonusTables](const MatchPairer::BotSpec& spec, Search* search) {
          assert(spec.botIdx < patternBonusTables.size());
          search->setCopyOfExternalPatternBonusTable(patternBonusTables[spec.botIdx]);
        };
        std::function<void(
          const Board&, const BoardHistory&, Player, Loc,
          const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
          const Search*
        )> onEachMove = nullptr;
        if(matchMoveOut != NULL) {
          onEachMove = [
            &matchMoveOut,&resultMutex,scheduleEntry,seed,botSpecB,botSpecW
          ](
            const Board& board,
            const BoardHistory& hist,
            Player pla,
            Loc moveLoc,
            const std::vector<double>& historicalWinLossValues,
            const std::vector<double>& historicalLeads,
            const std::vector<double>& historicalScoreStdevs,
            const Search* search
          ) {
            (void)historicalWinLossValues;
            (void)historicalLeads;
            (void)historicalScoreStdevs;
            ReportedSearchValues values = search->getRootValuesRequireSuccess();
            double perspectiveFactor = pla == P_WHITE ? 1.0 : -1.0;
            nlohmann::json moveRecord;
            moveRecord["schemaVersion"] = 1;
            if(scheduleEntry != NULL) {
              moveRecord["scheduleId"] = scheduleEntry->scheduleId;
              moveRecord["gameId"] = scheduleEntry->gameId;
              if(!scheduleEntry->pairId.empty())
                moveRecord["pairId"] = scheduleEntry->pairId;
              if(!scheduleEntry->positionId.empty())
                moveRecord["positionId"] = scheduleEntry->positionId;
            }
            else {
              moveRecord["scheduleId"] = nullptr;
              moveRecord["gameId"] = seed;
            }
            moveRecord["seed"] = seed;
            moveRecord["turnNumber"] = hist.getCurrentTurnNumber();
            moveRecord["player"] = pla == P_BLACK ? "B" : "W";
            moveRecord["bot"] = pla == P_BLACK ? botSpecB.botName : botSpecW.botName;
            moveRecord["move"] = Location::toString(moveLoc,board);
            moveRecord["winProbability"] = pla == P_WHITE ? values.winValue : values.lossValue;
            moveRecord["lossProbability"] = pla == P_WHITE ? values.lossValue : values.winValue;
            moveRecord["noResultProbability"] = values.noResultValue;
            moveRecord["scoreMean"] = values.expectedScore * perspectiveFactor;
            moveRecord["scoreStdev"] = values.expectedScoreStdev;
            moveRecord["scoreLead"] = values.lead * perspectiveFactor;
            moveRecord["resultUtility"] = values.resultUtility * perspectiveFactor;
            moveRecord["scoreUtility"] = values.scoreUtility * perspectiveFactor;
            moveRecord["otherUtility"] = values.otherUtility * perspectiveFactor;
            moveRecord["utility"] = values.utility * perspectiveFactor;
            moveRecord["lowerScoreTailProb"] =
              pla == P_WHITE ? values.lowerScoreTailProb : values.upperScoreTailProb;
            moveRecord["upperScoreTailProb"] =
              pla == P_WHITE ? values.upperScoreTailProb : values.lowerScoreTailProb;
            moveRecord["visits"] = values.visits;
            moveRecord["weight"] = values.weight;
            std::lock_guard<std::mutex> lock(resultMutex);
            (*matchMoveOut) << moveRecord.dump() << "\n";
          };
        }
        gameData = gameRunner->runGame(
          seed, botSpecB, botSpecW, NULL,
          scheduleEntry == NULL ? NULL : &scheduleEntry->startPosition,
          logger,
          shouldStopFunc, shouldPause, nullptr, afterInitialization, onEachMove
        );
      }

      bool shouldContinue = gameData != NULL;
      if(gameData != NULL) {
        if(scheduleEntry != NULL) {
          if(gameData->startHist.initialTurnNumber != scheduleEntry->startPosition.initialTurnNumber ||
             gameData->startHist.moveHistory.size() != scheduleEntry->startPosition.moves.size()) {
            throw StringError(
              "Deterministic game " + scheduleEntry->gameId +
              " did not replay its complete startPosition (check move legality and that the position is not terminal)"
            );
          }
        }
        if(sgfOut != NULL) {
          WriteSgf::writeSgf(*sgfOut,gameData->bName,gameData->wName,gameData->endHist,gameData,false,true);
          (*sgfOut) << endl;
        }
        if(matchResultOut != NULL) {
          nlohmann::json result = makeMatchResultJson(*gameData,seed,scheduleEntry);
          std::lock_guard<std::mutex> lock(resultMutex);
          (*matchResultOut) << result.dump() << "\n";
        }

        {
          std::lock_guard<std::mutex> lock(statsMutex);
          gameCount += 1;
          timeUsedByBotMap[gameData->bName] += gameData->bTimeUsed;
          timeUsedByBotMap[gameData->wName] += gameData->wTimeUsed;
          movesByBotMap[gameData->bName] += (double)gameData->bMoveCount;
          movesByBotMap[gameData->wName] += (double)gameData->wMoveCount;

          int64_t x = gameCount;
          while(x % 2 == 0 && x > 1) x /= 2;
          if(x == 1 || x == 3 || x == 5) {
            for(auto& pair : timeUsedByBotMap) {
              logger.write(
                "Avg move time used by " + pair.first + " " +
                Global::doubleToString(pair.second / movesByBotMap[pair.first]) + " " +
                Global::doubleToString(movesByBotMap[pair.first]) + " moves"
              );
            }
          }
        }

        delete gameData;
      }

      if(shouldStop.load())
        break;
      if(!shouldContinue)
        break;
    }
    if(sgfOut != NULL) {
      sgfOut->close();
      delete sgfOut;
    }
    logger.write("Match loop thread terminating");
  };
  auto runMatchLoopProtected = [&logger,&runMatchLoop](uint64_t threadHash) {
    Logger::logThreadUncaught("match loop", &logger, [&](){ runMatchLoop(threadHash); });
  };


  Rand hashRand;
  vector<std::thread> threads;
  threads.reserve(numGameThreads);
  for(int i = 0; i<numGameThreads; i++) {
    threads.emplace_back(runMatchLoopProtected, hashRand.nextUInt64());
  }
  for(int i = 0; i<threads.size(); i++)
    threads[i].join();

  if(matchResultOut != NULL) {
    matchResultOut->close();
    delete matchResultOut;
  }
  if(matchMoveOut != NULL) {
    matchMoveOut->close();
    delete matchMoveOut;
  }
  delete matchPairer;
  delete gameRunner;

  nnEvalsByBot.clear();
  for(int i = 0; i<nnEvals.size(); i++) {
    if(nnEvals[i] != NULL) {
      logger.write(nnEvals[i]->getModelFileName());
      logger.write("NN rows: " + Global::int64ToString(nnEvals[i]->numRowsProcessed()));
      logger.write("NN batches: " + Global::int64ToString(nnEvals[i]->numBatchesProcessed()));
      logger.write("NN avg batch size: " + Global::doubleToString(nnEvals[i]->averageProcessedBatchSize()));
      delete nnEvals[i];
    }
  }
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}
