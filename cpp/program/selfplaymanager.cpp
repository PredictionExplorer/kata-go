#include "../program/selfplaymanager.h"

#include "../core/test.h"

#include <chrono>

using namespace std;

ExtremeCohortGameBuffer::CohortBucket::CohortBucket()
  :gamesByAttempt(),
   failed(false),
   failureReason()
{}

ExtremeCohortGameBuffer::CohortBucket::CohortBucket(int groupSize)
  :gamesByAttempt(groupSize,NULL),
   failed(false),
   failureReason()
{}

bool ExtremeCohortGameBuffer::CohortKey::operator<(const CohortKey& other) const {
  return std::tie(
    runIdentity,cohortId,groupSize,focalPla,focalModelIdentity,opponentModelIdentity,configIdentity
  ) < std::tie(
    other.runIdentity,other.cohortId,other.groupSize,other.focalPla,
    other.focalModelIdentity,other.opponentModelIdentity,other.configIdentity
  );
}

ExtremeCohortGameBuffer::ExtremeCohortGameBuffer()
  :pending()
{}

ExtremeCohortGameBuffer::~ExtremeCohortGameBuffer() {
  discardIncomplete();
}

ExtremeCohortGameBuffer::AddResult ExtremeCohortGameBuffer::addGame(
  FinishedGameData* gameData,
  vector<FinishedGameData*>& completedGames,
  string& error
) {
  completedGames.clear();
  error.clear();
  if(gameData == NULL) {
    error = "null game submitted to cohort buffer";
    return COHORT_REJECTED;
  }
  const ExtremeCohortData& assignment = gameData->extremeCohort;
  if(!assignment.hasValidAssignment()) {
    error = "game has no valid extreme-cohort launch assignment";
    delete gameData;
    return COHORT_REJECTED;
  }

  CohortKey key;
  key.cohortId = assignment.cohortId;
  key.groupSize = assignment.groupSize;
  key.focalPla = assignment.focalPla;
  key.focalModelIdentity = assignment.focalModelIdentity;
  key.opponentModelIdentity = assignment.opponentModelIdentity;
  key.configIdentity = assignment.configIdentity;
  key.runIdentity = assignment.runIdentity;

  auto insertion = pending.emplace(key,CohortBucket(assignment.groupSize));
  CohortBucket& bucket = insertion.first->second;
  if(bucket.gamesByAttempt[assignment.attemptIdx] != NULL) {
    bucket.failed = true;
    bucket.failureReason = "duplicate attempt index received";
    error = bucket.failureReason;
    delete gameData;
    return COHORT_REJECTED;
  }
  bucket.gamesByAttempt[assignment.attemptIdx] = gameData;

  for(FinishedGameData* game: bucket.gamesByAttempt) {
    if(game == NULL)
      return BUFFERED;
  }

  vector<FinishedGameData*> games = std::move(bucket.gamesByAttempt);
  const bool bucketFailed = bucket.failed;
  const string bucketFailureReason = bucket.failureReason;
  pending.erase(insertion.first);

  if(bucketFailed) {
    error = bucketFailureReason;
    for(FinishedGameData* game: games)
      delete game;
    return COHORT_REJECTED;
  }
  if(!applyExtremeCohortCredits(games,error)) {
    for(FinishedGameData* game: games)
      delete game;
    return COHORT_REJECTED;
  }

  completedGames = std::move(games);
  return COHORT_READY;
}

size_t ExtremeCohortGameBuffer::discardIncomplete() {
  size_t numDiscarded = 0;
  for(auto& entry: pending) {
    for(FinishedGameData* game: entry.second.gamesByAttempt) {
      if(game != NULL) {
        delete game;
        numDiscarded += 1;
      }
    }
  }
  pending.clear();
  return numDiscarded;
}

size_t ExtremeCohortGameBuffer::numPendingGames() const {
  size_t numGames = 0;
  for(const auto& entry: pending) {
    for(const FinishedGameData* game: entry.second.gamesByAttempt) {
      if(game != NULL)
        numGames += 1;
    }
  }
  return numGames;
}

size_t ExtremeCohortGameBuffer::numPendingCohorts() const {
  return pending.size();
}

//------------------------------------------------------------------------------------

SelfplayManager::ModelData::ModelData(
  const string& name, NNEvaluator* neval, int maxDQueueSize,
  TrainingDataWriter* tdWriter, ofstream* sOut,
  double initialTime,
  bool hasDataLoop
):
  modelName(name),
  nnEval(neval),
  gameStartedCount(0),
  gamesFinishedCount(0),
  movesPlayedCount(0),
  lastReleaseTime(initialTime),
  hasDataWriteLoop(hasDataLoop),
  finishedGameQueue(maxDQueueSize),
  maxUnwrittenGames((size_t)maxDQueueSize),
  unwrittenGamesMutex(),
  unwrittenGamesCapacity(),
  numUnwrittenGames(0),
  dataQueueReadOnly(false),
  acquireCount(0),
  hasOpenExtremeCohort(false),
  openExtremeCohortId(0),
  nextExtremeAttemptIdx(0),
  tdataWriter(tdWriter),
  sgfOut(sOut)
{
}

bool SelfplayManager::ModelData::waitForUnwrittenGameCapacity(
  const std::function<bool()>& shouldStop
) {
  std::unique_lock<std::mutex> lock(unwrittenGamesMutex);
  while(!dataQueueReadOnly && numUnwrittenGames >= maxUnwrittenGames) {
    if(shouldStop != nullptr && shouldStop())
      return false;
    unwrittenGamesCapacity.wait_for(lock,std::chrono::milliseconds(100));
  }
  if(dataQueueReadOnly || (shouldStop != nullptr && shouldStop()))
    return false;
  numUnwrittenGames += 1;
  return true;
}

void SelfplayManager::ModelData::releaseUnwrittenGames(size_t count) {
  if(count == 0)
    return;
  std::lock_guard<std::mutex> lock(unwrittenGamesMutex);
  testAssert(count <= numUnwrittenGames);
  const bool wasFull = numUnwrittenGames >= maxUnwrittenGames;
  numUnwrittenGames -= count;
  if(wasFull || count > 1)
    unwrittenGamesCapacity.notify_all();
  else
    unwrittenGamesCapacity.notify_one();
}

void SelfplayManager::ModelData::setDataQueueReadOnly() {
  {
    std::lock_guard<std::mutex> lock(unwrittenGamesMutex);
    dataQueueReadOnly = true;
    unwrittenGamesCapacity.notify_all();
  }
  finishedGameQueue.setReadOnly();
}

size_t SelfplayManager::ModelData::getNumUnwrittenGames() {
  std::lock_guard<std::mutex> lock(unwrittenGamesMutex);
  return numUnwrittenGames;
}

SelfplayManager::ModelData::~ModelData() {
  delete nnEval;
  delete tdataWriter;
  if(sgfOut != NULL)
    delete sgfOut;
}

//------------------------------------------------------------------------------------

SelfplayManager::SelfplayManager(
  int maxDQueueSize,
  Logger* lg,
  int64_t logEvery,
  bool autoCleanup,
  const ExtremeCohortSettings& cohortSettings
):
  maxDataQueueSize(maxDQueueSize),
  logger(lg),
  logGamesEvery(logEvery),
  autoCleanupAllButLatestIfUnused(autoCleanup),
  extremeCohortSettings(cohortSettings),
  timer(),
  managerMutex(),
  modelDatas(),
  numDataWriteLoopsActive(0),
  dataWriteLoopsAreDone(),
  totalNumRowsProcessed(0),
  nextExtremeCohortId(cohortSettings.cohortIdStart())
{
  if(
    extremeCohortSettings.groupSize < 0 ||
    extremeCohortSettings.groupSize > 64 ||
    (extremeCohortSettings.isEnabled() &&
     (extremeCohortSettings.focalPla != P_BLACK && extremeCohortSettings.focalPla != P_WHITE))
  )
    throw StringError("SelfplayManager: invalid extreme cohort size or focal color");
  if(
    extremeCohortSettings.isEnabled() &&
    (maxDataQueueSize < extremeCohortSettings.groupSize ||
     maxDataQueueSize % extremeCohortSettings.groupSize != 0)
  )
    throw StringError("SelfplayManager: data queue size must be a multiple of cohort size");
  if(
    extremeCohortSettings.isEnabled() &&
    (extremeCohortSettings.configIdentity.empty() || extremeCohortSettings.runIdentity.empty())
  )
    throw StringError("SelfplayManager: extreme cohorts require config and run identities");
}

SelfplayManager::~SelfplayManager() {
  std::unique_lock<std::mutex> lock(managerMutex);
  for(size_t i = 0; i<modelDatas.size(); i++) {
    //If a client tries to delete this while something is still acquired, there's something wrong.
    testAssert(modelDatas[i]->acquireCount == 0);
    //Trigger data writing loop to quit once it reaches end of its queue
    modelDatas[i]->setDataQueueReadOnly();
    totalNumRowsProcessed += modelDatas[i]->nnEval->numRowsProcessed();
    //Data write loop is responsible for deleting ModelData, if it exists
    if(!modelDatas[i]->hasDataWriteLoop)
      delete modelDatas[i];
  }
  modelDatas.clear();
  while(numDataWriteLoopsActive > 0) {
    dataWriteLoopsAreDone.wait(lock);
  }
}

uint64_t SelfplayManager::getTotalNumRowsProcessed() const {
  std::lock_guard<std::mutex> lock(managerMutex);
  uint64_t total = totalNumRowsProcessed;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    total += modelDatas[i]->nnEval->numRowsProcessed();
  }
  return total;
}


static void dataWriteLoop(SelfplayManager* manager, SelfplayManager::ModelData* modelData) {
  manager->runDataWriteLoop(modelData);
}

void SelfplayManager::maybeAutoCleanupAlreadyLocked() {
  if(autoCleanupAllButLatestIfUnused && modelDatas.size() > 0) {
    for(size_t i = 0; i<modelDatas.size()-1; i++) {
      ModelData* foundData = modelDatas[i];
      if(foundData->acquireCount <= 0) {
        testAssert(foundData->acquireCount == 0);
        //Trigger data writing loop to quit once it reaches end of its queue
        foundData->setDataQueueReadOnly();
        totalNumRowsProcessed += foundData->nnEval->numRowsProcessed();
        //Data write loop is responsible for deleting ModelData, if it exists
        if(!foundData->hasDataWriteLoop)
          delete foundData;
        modelDatas.erase(modelDatas.begin()+i);
        i--;
      }
    }
  }
}


void SelfplayManager::cleanupUnusedModelsOlderThan(double seconds) {
  std::lock_guard<std::mutex> lock(managerMutex);
  double now = timer.getSeconds();
  for(size_t i = 0; i<modelDatas.size(); i++) {
    ModelData* foundData = modelDatas[i];
    if(foundData->acquireCount <= 0 && now - foundData->lastReleaseTime > seconds) {
      testAssert(foundData->acquireCount == 0);
      logger->write("Unloading network that hasn't been used in a while: " + foundData->modelName);
      //Trigger data writing loop to quit once it reaches end of its queue
      foundData->setDataQueueReadOnly();
      totalNumRowsProcessed += foundData->nnEval->numRowsProcessed();
      //Data write loop is responsible for deleting ModelData, if it exists
      if(!foundData->hasDataWriteLoop)
        delete foundData;
      modelDatas.erase(modelDatas.begin()+i);
      i--;
    }
  }
}

void SelfplayManager::clearUnusedModelCaches() {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i<modelDatas.size(); i++) {
    ModelData* foundData = modelDatas[i];
    if(foundData->acquireCount <= 0) {
      foundData->nnEval->clearCache();
    }
  }
}


void SelfplayManager::loadModelAndStartDataWriting(
  NNEvaluator* nnEval,
  TrainingDataWriter* tdataWriter,
  ofstream* sgfOut
) {
  string modelName = nnEval->getModelName();
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      throw StringError("SelfplayManager::loadModelAndStartDataWriting: Duplicate model name: " + modelName);
    }
  }

  double initialTime = timer.getSeconds();
  bool hasDataWriteLoop = true;
  ModelData* newModel = new ModelData(modelName,nnEval,maxDataQueueSize,tdataWriter,sgfOut,initialTime,hasDataWriteLoop);
  modelDatas.push_back(newModel);
  numDataWriteLoopsActive++;
  std::thread newThread(dataWriteLoop,this,newModel);
  newThread.detach();

  maybeAutoCleanupAlreadyLocked();
}

void SelfplayManager::loadModelNoDataWritingLoop(
  NNEvaluator* nnEval,
  TrainingDataWriter* tdataWriter,
  ofstream* sgfOut
) {
  string modelName = nnEval->getModelName();
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      throw StringError("SelfplayManager::loadModelAndStartDataWriting: Duplicate model name: " + modelName);
    }
  }

  double initialTime = timer.getSeconds();
  bool hasDataWriteLoop = false;
  ModelData* newModel = new ModelData(modelName,nnEval,maxDataQueueSize,tdataWriter,sgfOut,initialTime,hasDataWriteLoop);
  modelDatas.push_back(newModel);
  maybeAutoCleanupAlreadyLocked();
}

size_t SelfplayManager::numModels() const {
  std::lock_guard<std::mutex> lock(managerMutex);
  return modelDatas.size();
}

vector<string> SelfplayManager::modelNames() const {
  std::lock_guard<std::mutex> lock(managerMutex);
  vector<string> names;
  names.reserve(modelDatas.size());
  for(size_t i = 0; i<modelDatas.size(); i++)
    names.push_back(modelDatas[i]->modelName);
  return names;
}

string SelfplayManager::getLatestModelName() const {
  std::lock_guard<std::mutex> lock(managerMutex);
  if(modelDatas.size() <= 0)
    throw StringError("SelfplayManager::getLatestModelName: no models loaded");
  return modelDatas[modelDatas.size()-1]->modelName;
}

bool SelfplayManager::hasModel(const std::string& modelName) const {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName)
      return true;
  }
  return false;
}


NNEvaluator* SelfplayManager::acquireModelAlreadyLocked(ModelData* foundData) {
  foundData->acquireCount += 1;
  return foundData->nnEval;
}
void SelfplayManager::releaseAlreadyLocked(ModelData* foundData) {
  foundData->lastReleaseTime = timer.getSeconds();
  foundData->acquireCount -= 1;
}

NNEvaluator* SelfplayManager::acquireModel(const string& modelName) {
  std::lock_guard<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData != NULL)
    return acquireModelAlreadyLocked(foundData);
  return NULL;
}

NNEvaluator* SelfplayManager::acquireLatest() {
  std::lock_guard<std::mutex> lock(managerMutex);
  if(modelDatas.size() <= 0)
    return NULL;
  ModelData* foundData = modelDatas[modelDatas.size()-1];
  return acquireModelAlreadyLocked(foundData);
}

void SelfplayManager::release(const string& modelName) {
  std::lock_guard<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData != NULL) {
    releaseAlreadyLocked(foundData);
    maybeAutoCleanupAlreadyLocked();
  }
}

void SelfplayManager::release(NNEvaluator* nnEval) {
  std::lock_guard<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData != NULL) {
    releaseAlreadyLocked(foundData);
    maybeAutoCleanupAlreadyLocked();
  }
}

void SelfplayManager::countOneGameStarted(NNEvaluator* nnEval) {
  (void)countOneGameStartedInternal(nnEval,"",false);
}

ExtremeCohortData SelfplayManager::countOneGameStartedAndGetCohort(
  NNEvaluator* nnEval,
  const string& opponentModelIdentity
) {
  return countOneGameStartedInternal(nnEval,opponentModelIdentity,true);
}

ExtremeCohortData SelfplayManager::countOneGameStartedInternal(
  NNEvaluator* nnEval,
  const string& opponentModelIdentity,
  bool assignExtremeCohort
) {
  std::unique_lock<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData == NULL)
    throw StringError("SelfplayManager::countOneGameStarted: could not find model. Possible bug - client did not acquire model?");
  if(extremeCohortSettings.isEnabled() && !assignExtremeCohort)
    throw StringError("SelfplayManager::countOneGameStarted: enabled extreme cohorts require launch assignments");
  if(extremeCohortSettings.isEnabled() && opponentModelIdentity.empty())
    throw StringError("SelfplayManager::countOneGameStarted: missing frozen opponent identity");

  foundData->gameStartedCount += 1;
  int64_t gameStartedCount = foundData->gameStartedCount;

  ExtremeCohortData assignment;
  if(extremeCohortSettings.isEnabled()) {
    if(!foundData->hasOpenExtremeCohort) {
      foundData->hasOpenExtremeCohort = true;
      foundData->openExtremeCohortId = nextExtremeCohortId;
      nextExtremeCohortId += 1;
      foundData->nextExtremeAttemptIdx = 0;
    }

    assignment.enabled = true;
    assignment.metadataVersion = ExtremeCohortData::METADATA_VERSION;
    assignment.cohortId = foundData->openExtremeCohortId;
    assignment.attemptIdx = foundData->nextExtremeAttemptIdx;
    assignment.groupSize = extremeCohortSettings.groupSize;
    assignment.focalPla = extremeCohortSettings.focalPla;
    assignment.focalModelIdentity = nnEval->getModelName();
    assignment.opponentModelIdentity = opponentModelIdentity;
    assignment.configIdentity = extremeCohortSettings.configIdentity;
    assignment.runIdentity = extremeCohortSettings.runIdentity;
    testAssert(assignment.hasValidAssignment());

    foundData->nextExtremeAttemptIdx += 1;
    if(foundData->nextExtremeAttemptIdx >= extremeCohortSettings.groupSize) {
      testAssert(foundData->nextExtremeAttemptIdx == extremeCohortSettings.groupSize);
      foundData->hasOpenExtremeCohort = false;
      foundData->nextExtremeAttemptIdx = 0;
    }
  }
  lock.unlock();

  if(logger != NULL && gameStartedCount % logGamesEvery == 0) {
    logger->write("Started " + Global::int64ToString(gameStartedCount) + " games with " + nnEval->getModelName());
  }
  int64_t logNNEvery = logGamesEvery*100 > 1000 ? logGamesEvery*100 : 1000;
  if(logger != NULL && gameStartedCount % logNNEvery == 0) {
    logger->write(nnEval->getModelFileName());
    logger->write("Games finished: " + Global::int64ToString(foundData->gamesFinishedCount.load(std::memory_order_relaxed)));
    logger->write("Moves played: " + Global::int64ToString(foundData->movesPlayedCount.load(std::memory_order_relaxed)));
    if(foundData->tdataWriter != NULL)
      logger->write("Data rows: " + Global::int64ToString(foundData->tdataWriter->numRowsWritten()));
    logger->write("NN rows: " + Global::int64ToString(nnEval->numRowsProcessed()));
    logger->write("NN batches: " + Global::int64ToString(nnEval->numBatchesProcessed()));
    logger->write("NN avg batch size: " + Global::doubleToString(nnEval->averageProcessedBatchSize()));
    logger->write("NN cache hits: " + Global::int64ToString((int64_t)nnEval->numCacheHits()));
  }
  return assignment;
}

bool SelfplayManager::waitForDataToWriteCapacity(
  NNEvaluator* nnEval,
  const std::function<bool()>& shouldStop
) {
  if(!extremeCohortSettings.isEnabled())
    return true;
  std::unique_lock<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData == NULL)
    throw StringError("SelfplayManager::waitForDataToWriteCapacity: could not find acquired model");
  lock.unlock();
  return foundData->waitForUnwrittenGameCapacity(shouldStop);
}

void SelfplayManager::cancelDataToWriteReservation(NNEvaluator* nnEval) {
  if(!extremeCohortSettings.isEnabled())
    return;
  std::unique_lock<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData == NULL)
    throw StringError("SelfplayManager::cancelDataToWriteReservation: could not find acquired model");
  lock.unlock();
  foundData->releaseUnwrittenGames(1);
}

void SelfplayManager::enqueueDataToWrite(const string& modelName, FinishedGameData* gameData) {
  std::unique_lock<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData == NULL)
    throw StringError("SelfplayManager::enqueueDataToWrite: could not find model. Possible bug - client did not acquire model?");
  testAssert(foundData->hasDataWriteLoop == true);

  //In case it takes a while to push the game on, drop the lock. We're guaranteed as a precondition that
  //the caller has acquired the model as well, so it won't be cleaned up underneath us.
  lock.unlock();
  if(!foundData->finishedGameQueue.waitPush(gameData)) {
    if(extremeCohortSettings.isEnabled())
      foundData->releaseUnwrittenGames(1);
    delete gameData;
  }
}

void SelfplayManager::enqueueDataToWrite(NNEvaluator* nnEval, FinishedGameData* gameData) {
  std::unique_lock<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData == NULL)
    throw StringError("SelfplayManager::enqueueDataToWrite: could not find model. Possible bug - client did not acquire model?");

  //In case it takes a while to push the game on, drop the lock. We're guaranteed as a precondition that
  //the caller has acquired the model as well, so it won't be cleaned up underneath us.
  lock.unlock();
  if(!foundData->finishedGameQueue.waitPush(gameData)) {
    if(extremeCohortSettings.isEnabled())
      foundData->releaseUnwrittenGames(1);
    delete gameData;
  }
}

void SelfplayManager::runDataWriteLoop(ModelData* modelData) {
  Logger::logThreadUncaught("data write loop", logger, [&](){ runDataWriteLoopImpl(modelData); });
}

void SelfplayManager::runDataWriteLoopImpl(ModelData* modelData) {
  if(logger != NULL)
    logger->write("Data write loop starting for neural net: " + modelData->modelName);

  ExtremeCohortGameBuffer cohortBuffer;
  auto countFinishedGame = [&](const FinishedGameData* gameData) {
    modelData->gamesFinishedCount.fetch_add(1, std::memory_order_relaxed);
    // Moves actually played by search this game (excludes any pre-placed opening/start-position moves).
    testAssert(gameData->startHist.moveHistory.size() <= gameData->endHist.moveHistory.size());
    modelData->movesPlayedCount.fetch_add(
      (int64_t)(gameData->endHist.moveHistory.size() - gameData->startHist.moveHistory.size()),
      std::memory_order_relaxed
    );
  };
  auto writeGame = [&](FinishedGameData* gameData, bool countAfterDataWrite) {
    modelData->tdataWriter->writeGame(*gameData);
    if(countAfterDataWrite)
      countFinishedGame(gameData);
    if(modelData->sgfOut != NULL) {
      testAssert(gameData->startHist.moveHistory.size() <= gameData->endHist.moveHistory.size());
      WriteSgf::writeSgf(*modelData->sgfOut,gameData->bName,gameData->wName,gameData->endHist,gameData,false,true);
      (*modelData->sgfOut) << endl;
    }
    delete gameData;
    if(extremeCohortSettings.isEnabled())
      modelData->releaseUnwrittenGames(1);
  };

  while(true) {
    size_t size =
      extremeCohortSettings.isEnabled() ?
      modelData->getNumUnwrittenGames() :
      modelData->finishedGameQueue.size();
    if(size > maxDataQueueSize / 2 && logger != NULL)
      logger->write(Global::strprintf("WARNING: Struggling to keep up writing data, %d games pending out of %d max",size,maxDataQueueSize));

    FinishedGameData* gameData;
    bool suc = modelData->finishedGameQueue.waitPop(gameData);
    if(!suc)
      break;

    testAssert(gameData != NULL);

    if(!extremeCohortSettings.isEnabled()) {
      //Keep ordinary-mode ordering unchanged: write data, count, write SGF, delete.
      writeGame(gameData,true);
      continue;
    }

    //Extreme cohorts may remain buffered or be discarded at shutdown, but these games did
    //finish, so account for them when they leave the completion queue.
    countFinishedGame(gameData);

    const ExtremeCohortData& assignment = gameData->extremeCohort;
    if(
      !assignment.hasValidAssignment() ||
      assignment.groupSize != extremeCohortSettings.groupSize ||
      assignment.focalPla != extremeCohortSettings.focalPla ||
      assignment.focalModelIdentity != modelData->modelName ||
      assignment.configIdentity != extremeCohortSettings.configIdentity ||
      assignment.runIdentity != extremeCohortSettings.runIdentity
    ) {
      if(logger != NULL)
        logger->write("WARNING: Dropping game with an invalid or mismatched extreme cohort assignment for " + modelData->modelName);
      delete gameData;
      modelData->releaseUnwrittenGames(1);
      continue;
    }

    vector<FinishedGameData*> completedGames;
    string cohortError;
    const size_t pendingBefore = cohortBuffer.numPendingGames();
    ExtremeCohortGameBuffer::AddResult result =
      cohortBuffer.addGame(gameData,completedGames,cohortError);
    const size_t pendingAfter = cohortBuffer.numPendingGames();
    testAssert(pendingBefore + 1 >= pendingAfter + completedGames.size());
    const size_t numDeleted =
      pendingBefore + 1 - pendingAfter - completedGames.size();
    modelData->releaseUnwrittenGames(numDeleted);
    if(result == ExtremeCohortGameBuffer::COHORT_REJECTED) {
      if(logger != NULL)
        logger->write("WARNING: Dropping invalid extreme cohort for " + modelData->modelName + ": " + cohortError);
      continue;
    }
    if(result == ExtremeCohortGameBuffer::COHORT_READY) {
      //addGame returns attempt-index order, independent of completion order.
      for(FinishedGameData* completedGame: completedGames)
        writeGame(completedGame,false);
    }
  }

  size_t numIncompleteGames = cohortBuffer.discardIncomplete();
  modelData->releaseUnwrittenGames(numIncompleteGames);
  if(numIncompleteGames > 0 && logger != NULL) {
    logger->write(
      "WARNING: Dropped " + Global::uint64ToString((uint64_t)numIncompleteGames) +
      " finished games from incomplete extreme cohorts for " + modelData->modelName
    );
  }

  modelData->tdataWriter->flushIfNonempty();
  if(modelData->sgfOut != NULL)
    modelData->sgfOut->close();

  if(logger != NULL)
    logger->write("Data write loop finishing for neural net: " + modelData->modelName);

  if(extremeCohortSettings.isEnabled())
    testAssert(modelData->getNumUnwrittenGames() == 0);
  testAssert(modelData->acquireCount == 0);

  string name = modelData->modelName;

  //Lock the manager and do nothing with the lock (except run an assert).
  //The lock is technically necessary for thread-safety - we don't want to delete this modelData until we are
  //absolutely sure that the manager is done removing it from its own tracking in modelDatas, so we lock
  //the manager to make sure that we block until this is the case. While we're at it, we go ahead and assert it too.
  {
    std::lock_guard<std::mutex> lock(managerMutex);
    for(size_t i = 0; i<modelDatas.size(); i++) {
      (void)i;
      testAssert(modelDatas[i] != modelData);
    }
  }

  //Do logging and cleanup while unlocked, so that our freeing and stopping of this neural net doesn't
  //block anyone else
  if(logger != NULL) {
    logger->write("Final cleanup of net: " + modelData->nnEval->getModelFileName());
    logger->write("Final games finished: " + Global::int64ToString(modelData->gamesFinishedCount.load(std::memory_order_relaxed)));
    logger->write("Final moves played: " + Global::int64ToString(modelData->movesPlayedCount.load(std::memory_order_relaxed)));
    logger->write("Final data rows: " + Global::int64ToString(modelData->tdataWriter->numRowsWritten()));
    logger->write("Final NN rows: " + Global::int64ToString(modelData->nnEval->numRowsProcessed()));
    logger->write("Final NN batches: " + Global::int64ToString(modelData->nnEval->numBatchesProcessed()));
    logger->write("Final NN avg batch size: " + Global::doubleToString(modelData->nnEval->averageProcessedBatchSize()));
    logger->write("Final NN cache hits: " + Global::int64ToString((int64_t)modelData->nnEval->numCacheHits()));
  }

  delete modelData;

  if(logger != NULL) {
    logger->write("Data write loop cleaned up and terminating for " + name);
  }

  //Check back in and notify that we're done once done cleaning up.
  std::unique_lock<std::mutex> lock(managerMutex);
  numDataWriteLoopsActive--;
  testAssert(numDataWriteLoopsActive >= 0);
  if(numDataWriteLoopsActive == 0) {
    testAssert(modelDatas.size() == 0);
    dataWriteLoopsAreDone.notify_all();
  }
  lock.unlock();
}

void SelfplayManager::withDataWriters(
  NNEvaluator* nnEval,
  const std::function<void(TrainingDataWriter* tdataWriter, std::ofstream* sgfOut)>& f
) {
  std::lock_guard<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData == NULL)
    throw StringError("SelfplayManager::withDataWriters: could not find model. Possible bug - client did not acquire model?");
  testAssert(foundData->hasDataWriteLoop == false);

  f(foundData->tdataWriter, foundData->sgfOut);
}
