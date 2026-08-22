#ifndef PROGRAM_SELFPLAYMANAGER_H_
#define PROGRAM_SELFPLAYMANAGER_H_

#include <atomic>
#include <condition_variable>
#include <map>
#include <tuple>

#include "../core/threadsafequeue.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"
#include "../neuralnet/nneval.h"

//Single-threaded completion buffer used by a model's data-write loop. It owns every
//game passed to addGame until either a complete validated cohort is returned or the
//game is discarded. The identity tuple is part of the key, so generations, colors,
//runs, and configs can never complete one another's cohorts.
class ExtremeCohortGameBuffer {
 public:
  enum AddResult {
    BUFFERED,
    COHORT_READY,
    COHORT_REJECTED
  };

  ExtremeCohortGameBuffer();
  ~ExtremeCohortGameBuffer();

  ExtremeCohortGameBuffer(const ExtremeCohortGameBuffer&) = delete;
  ExtremeCohortGameBuffer& operator=(const ExtremeCohortGameBuffer&) = delete;

  AddResult addGame(
    FinishedGameData* gameData,
    std::vector<FinishedGameData*>& completedGames,
    std::string& error
  );

  //Delete all incomplete cohorts. Returns the number of games discarded.
  size_t discardIncomplete();
  size_t numPendingGames() const;
  size_t numPendingCohorts() const;

 private:
  struct CohortKey {
    uint64_t cohortId;
    int groupSize;
    Player focalPla;
    std::string focalModelIdentity;
    std::string opponentModelIdentity;
    std::string configIdentity;
    std::string runIdentity;

    bool operator<(const CohortKey& other) const;
  };

  struct CohortBucket {
    std::vector<FinishedGameData*> gamesByAttempt;
    bool failed;
    std::string failureReason;

    CohortBucket();
    explicit CohortBucket(int groupSize);
  };

  std::map<CohortKey,CohortBucket> pending;
};

class SelfplayManager {
 public:
  SelfplayManager(
    int maxDataQueueSize,
    Logger* logger,
    int64_t logGamesEvery,
    bool autoCleanupAllButLatestIfUnused,
    const ExtremeCohortSettings& extremeCohortSettings = ExtremeCohortSettings()
  );
  ~SelfplayManager();

  SelfplayManager(const SelfplayManager& other);
  SelfplayManager& operator=(const SelfplayManager& other);
  SelfplayManager(SelfplayManager&& other);
  SelfplayManager& operator=(SelfplayManager&& other);

  //All below functions are internally synchronized and thread-safe.

  //SelfplayManager takes responsibility for deleting the data writers and closing and deleting sgfOut.
  //loadModelNoDataWritingLoop is for the manual writing interface
  void loadModelAndStartDataWriting(
    NNEvaluator* nnEval,
    TrainingDataWriter* tdataWriter,
    std::ofstream* sgfOut
  );
  void loadModelNoDataWritingLoop(
    NNEvaluator* nnEval,
    TrainingDataWriter* tdataWriter,
    std::ofstream* sgfOut
  );

  //NN queries summed across all the models managed by this manager over all time.
  uint64_t getTotalNumRowsProcessed() const;

  //For all of the below, model names are simply from nnEval->getModelName().

  //Models that aren't cleaned up yet are in the order from earliest to latest
  std::vector<std::string> modelNames() const;
  std::string getLatestModelName() const;
  bool hasModel(const std::string& modelName) const;
  size_t numModels() const;

  //Returns NULL if acquire failed (such as if that model was scheduled to be cleaned up or already cleaned up,).
  //Must call release when done, and cease using the NNEvaluator after that.
  NNEvaluator* acquireModel(const std::string& modelName);
  NNEvaluator* acquireLatest();
  //Release a model either by name or by the nnEval object that was returned.
  void release(const std::string& modelName);
  void release(NNEvaluator* nnEval);

  //Clean up any currently-unused models if their last usage was older than this many seconds ago.
  void cleanupUnusedModelsOlderThan(double seconds);
  //Clear the evaluation caches of any models that are currently unused.
  void clearUnusedModelCaches();

  //====================================================================================
  //These should only be called by a thread that has currently acquired the model.

  //Increment a counter and maybe log some stats
  void countOneGameStarted(NNEvaluator* nnEval);
  //The returned assignment is fixed before launch. In ordinary mode it is disabled.
  ExtremeCohortData countOneGameStartedAndGetCohort(
    NNEvaluator* nnEval,
    const std::string& opponentModelIdentity
  );

  // Extreme mode reserves bounded unwritten-data capacity before a game
  // starts, so in-flight, queued, and cohort-buffered games share one limit.
  bool waitForDataToWriteCapacity(
    NNEvaluator* nnEval,
    const std::function<bool()>& shouldStop
  );
  void cancelDataToWriteReservation(NNEvaluator* nnEval);

  //SelfplayManager takes responsibility for deleting the gameData once written.
  //Use these only if loadModelAndStartDataWriting was used to start the model.
  void enqueueDataToWrite(const std::string& modelName, FinishedGameData* gameData);
  void enqueueDataToWrite(NNEvaluator* nnEval, FinishedGameData* gameData);

  //Use these if loadModelNoDataWritingLoop was used to start the model.
  void withDataWriters(
    NNEvaluator* nnEval,
    const std::function<void(TrainingDataWriter* tdataWriter, std::ofstream* sgfOut)>& f
  );

  //====================================================================================

  //For internal use
  struct ModelData {
    std::string modelName;
    NNEvaluator* nnEval;
    int64_t gameStartedCount;
    // Counted at game-finish in the data write loop (lock-free), read cross-thread for logging.
    std::atomic<int64_t> gamesFinishedCount;
    std::atomic<int64_t> movesPlayedCount;
    double lastReleaseTime;
    bool hasDataWriteLoop;

    ThreadSafeQueue<FinishedGameData*> finishedGameQueue;
    const size_t maxUnwrittenGames;
    std::mutex unwrittenGamesMutex;
    std::condition_variable unwrittenGamesCapacity;
    size_t numUnwrittenGames;
    bool dataQueueReadOnly;
    int acquireCount;

    bool hasOpenExtremeCohort;
    uint64_t openExtremeCohortId;
    int nextExtremeAttemptIdx;

    TrainingDataWriter* tdataWriter;
    std::ofstream* sgfOut;

    ModelData(
      const std::string& name, NNEvaluator* neval, int maxDataQueueSize,
      TrainingDataWriter* tdWriter, std::ofstream* sOut,
      double initialLastReleaseTime,
      bool hasDataWriteLoop
    );
    ~ModelData();

    bool waitForUnwrittenGameCapacity(const std::function<bool()>& shouldStop);
    void releaseUnwrittenGames(size_t count);
    void setDataQueueReadOnly();
    size_t getNumUnwrittenGames();
  };

 private:
  const int maxDataQueueSize;
  Logger* logger;
  const int64_t logGamesEvery;
  const bool autoCleanupAllButLatestIfUnused;
  const ExtremeCohortSettings extremeCohortSettings;

  const ClockTimer timer;

  mutable std::mutex managerMutex;
  std::vector<ModelData*> modelDatas;
  int numDataWriteLoopsActive;
  std::condition_variable dataWriteLoopsAreDone;

  uint64_t totalNumRowsProcessed;
  uint64_t nextExtremeCohortId;

  NNEvaluator* acquireModelAlreadyLocked(SelfplayManager::ModelData* foundData);
  void releaseAlreadyLocked(SelfplayManager::ModelData* foundData);
  void maybeAutoCleanupAlreadyLocked();
  ExtremeCohortData countOneGameStartedInternal(
    NNEvaluator* nnEval,
    const std::string& opponentModelIdentity,
    bool assignExtremeCohort
  );
  void runDataWriteLoopImpl(ModelData* modelData);

 public:
  //For internal use
  void runDataWriteLoop(ModelData* modelData);

};

#endif //PROGRAM_SELFPLAYMANAGER_H_
