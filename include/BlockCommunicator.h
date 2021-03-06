#ifndef BlockCommunicator_h
#define BlockCommunicator_h
#include <assert.h>
#include "asserts.h"
#include "debug.h" // temporary
#include "aligned_vector.h"
#include "MPIdata.h"
#include "Parameters.h"
#include <list>
#include <mpi.h> // is there a way to forward declare mpi types?

// The combination of group (comm), tag, and neighbor
// should be unique for each connection.
//
// In a Cartesian topology, the challenges to this uniqueness
// occur for periodic boundary conditions when there is a
// dimension only 1 or 2 processes thick.
//
// In the 1-process-thick case, we can forgo MPI communication in
// that direction altogether.  But in the 2-process-thick case,
// we need to make a distinction between upward and downward
// channels of communication.  Distinguishing upward and downward
// directions is sufficient for any topology based on convex MPI
// subdomains, since in this case two subdomains can share at
// most two faces.  Any mechanism for distinguishing upward and
// downward directions for a pair of subdomains is sufficient;
// one way is to compare the positions of the centroids, first in
// the x direction, then in the y direction, and then in the z
// direction.  One might use tag=1 for downward communication and
// tag=2 for upward communication, and use MPI_COMM_WORLD for the
// group.
//
namespace Direction
{
  enum Enum
  {
    DEFAULT = 0,
    PARTICLE_DN, // downward communication of particles
    PARTICLE_UP, // upward communication of particles
    XDN,
    XUP,
    YDN,
    YUP,
    ZDN,
    ZUP
  };
}
class Connection
{
 private: // data
  // In MPI a message envelope includes the following
  // information plus the rank of this process.
  int _rank; // rank within group of neighbors we're connecting to
  int _tag; // tag to attach to messages
  MPI_Comm _comm; // communicator group
 public: // init
  // construct a connection that creates self-communication
  // in place of null connection
  static Connection null2self(int rank_, int tag_, int self_tag, MPI_Comm comm_)
  {
    Connection c(rank_,tag_,comm_);
    if(c._rank==MPI_PROC_NULL)
    {
      c._rank = MPIdata::get_rank();
      c._tag = self_tag;
    }
    return c;
  }
  Connection():
    _rank(0),
    _tag(0),
    _comm(MPI_COMM_WORLD)
  {}
  Connection(int rank_, int tag_, MPI_Comm comm_ = MPI_COMM_WORLD)
  {
    init(rank_,tag_,comm_);
  }
 private:
  void init(int rank_, int tag_, MPI_Comm comm_ = MPI_COMM_WORLD)
  {
    _rank = rank_;
    _tag = tag_;
    _comm = comm_;
  }
 public: // accessors
  int rank()const{return _rank;}
  int tag()const{return _tag;}
  MPI_Comm comm()const{return _comm;}
  static char* tag_name(int tag)
  {
    switch(tag)
    {
      default: unsupported_value_error(tag);
      case Direction::XDN: return "XDN";
      case Direction::XUP: return "XUP";
      case Direction::YDN: return "YDN";
      case Direction::YUP: return "YUP";
      case Direction::ZDN: return "ZDN";
      case Direction::ZUP: return "ZUP";
    }
  }
  const char* tag_name()const
  {
    return tag_name(_tag);
  }
};

inline bool signal_hack()
{
  return false;
}

// block of elements
template <class type>
struct Block
{
 private: // initialized in constructor
  // change the block to use an aligned allocator
  aligned_vector(type) block;
  int capacity;
  int listID;
  // used for MPI communication
  MPI_Request request;
  // hack to piggy-back information onto message
  int signal; // char
 private: // initialized at compile time
  // assumes using MPI_DOUBLE
  static const int NUMBERS_PER_ELEMENT = sizeof(type)/sizeof(double);
 public:
  int get_capacity()const{return capacity;}
 public:
  Block(int capacity_, int id_):
    capacity(capacity_),
    listID(id_),
    request(MPI_REQUEST_NULL),
    signal(0)
  {
    block.reserve(capacity);
  }
  // cancel any pending communication 
  ~Block()
  {
    if(request!=MPI_REQUEST_NULL)
    {
      MPI_Cancel(&request);
      MPI_Request_free(&request);
    }
  }
 public: // accessors
  MPI_Request& fetch_request(){return request;}
  aligned_vector(type)& fetch_block(){return block;}
  const aligned_vector(type)& get_block()const{return block;}
  int get_id(){ return listID; }
  void set_finished_flag() { signal |= 2; }
  void unset_finished_flag() { signal &= ~2; }
  bool finished_flag_is_set() { return signal & 2; }
  void set_insert_flag() { signal |= 1; }
  bool insert_flag_is_set() { return signal & 1; }
  int size(){return block.size();}

 public: // operations
  // used when the connection is null
  void increase_capacity()
  {
    capacity*=2;
    block.reserve(capacity);
  }
  // returns true if communication is complete; else returns false
  bool test_comm(MPI_Status& status)
  {
    int flag=0;
    MPI_Test(&request, &flag, &status);
    if(!flag)
      return false;
    // MPI_Test man page says this should now be true
    assert(request==MPI_REQUEST_NULL);
    return true;
  }
  bool test_comm()
  {
    MPI_Status status;
    return test_comm(status);
  }

 // sending
 //
 public: // sending
  type& operator[](int i){return block[i];}
  bool is_active()
  {
    return request!=MPI_REQUEST_NULL;
  }
  bool is_inactive()
  {
    return request==MPI_REQUEST_NULL;
  }
  bool isfull()
  {
    bool retval = (block.size()>=capacity);
    return retval;
  }
  // to implement this in the case that block is of type std::vector,
  // I would need to make the "size" a member of this class
  // and treat block.size() as a capacity.
  void fast_push_back(const type& in)
  {
    block.push_back(in);
    // assumes block is of type Larray
    //block.fast_push_back(in);
  }
  void push_back(const type& in)
  {
    block.push_back(in);
  }
  void send_block(const Connection& dest, int commID)
  {
    const int nop = size();
    assert_le(nop, capacity);
    bool last_block = (nop < capacity);
    // hack: put signal in an extra "particle"
    if(signal_hack())
    {
      type extra_element;
      double* extra_arr = (double*) &extra_element;
      assert(sizeof(type) >= sizeof(double)*2);
      // convert signal to double and put in extra_element
      extra_arr[0] = double(signal);
      // convert block ID to double and put in extra_element
      extra_arr[1] = double(commID);
      push_back(extra_element);
    }

    //dprintf("sending%s block (listID %d), (commID %d) to %d.%s of size %d",
    //  (last_block ? " final" : ""),
    //  listID, commID,
    //  dest.rank(), dest.tag_name(),
    //  nop);
    MPI_Isend(&block[0], NUMBERS_PER_ELEMENT*block.size(), MPI_DOUBLE,
      dest.rank(), dest.tag(), dest.comm(), &request);
    //dprintf("finished sending block number %d", listID);
  }
  void waitfor_send()
  {
    MPI_Status status;
    MPI_Wait(&request, &status);
  }
  void set_request(MPI_Request request_)
  {
    request = request_;
  }
  void clear()
  {
    block.clear();
    request = MPI_REQUEST_NULL;
    signal = 0;
  }
 public: // receiving
  // post a receive
  void recv_block(const Connection& source)
  {
    // verify that there is no receive already posted
    assert(request==MPI_REQUEST_NULL);
    signal=0;
    // make sure that space exists to receive
    int newsize = signal_hack() ? capacity+1 : capacity;
    block.resize(newsize);
    MPI_Irecv(&block[0], NUMBERS_PER_ELEMENT*block.size(), MPI_DOUBLE,
      source.rank(), source.tag(), source.comm(), &request);
  }
  // processing received data
  //
  // returns true if number of elements received
  // was smaller than capacity
  bool shrink_received_block(MPI_Status& status)
  {
    int count;
    int commID = -1;
    MPI_Get_count(&status, MPI_DOUBLE, &count);
    int num_elements_received = count / NUMBERS_PER_ELEMENT;
    // hack: handle the signal particle
    if(signal_hack())
    {
      // discount signal particle
      num_elements_received--;
      // extract signal from signal element
      double* extra_arr = (double*) &block[num_elements_received];
      signal = int(extra_arr[0]);
      commID = int(extra_arr[1]);
    }
    else
    {
      if(num_elements_received < capacity)
      {
        //dprintf("setting finished flag");
        set_finished_flag();
      }
      else
      {
        //unset_finished_flag();
      }
    }
    bool retval = (num_elements_received < capacity);
    assert_le(num_elements_received,capacity);
    // shrink to eliminate the trailing garbage
    block.resize(num_elements_received);
    //dprintf("received%s block (listID %d), (commID %d) of size %d",
    //  (finished_flag_is_set() ? " final" : ""),
    //  listID, commID,
    //  block.size());
    return retval;
  }
  void waitfor_recv()
  {
    MPI_Status status;
    MPI_Wait(&request, &status);
    shrink_received_block(status);
  }
  // returns true if message has been received.
};

// This could be separated into separate classes for sending and
// receiving.
//
// In the cases where the sender and receiver are the same
// process, it might be better to transfer data with a simple
// list.
//
// The implementation should probably be changed to use
// persistent requests.  My analysis of the decision 
// regarding message size is as follows:
//
//   Communication time is a linear combination of total data
//   to be sent and the number of messages it is broken into:
//  
//     time = data_time + message_overhead, i.e.,
//     time = data_size*data_coef + num_msgs*msg_coef.
//  
//   Assuming no blocking issues,
//   data_time is presumably the same for both persistent
//   and non-persistent communication.
//  
//   message_overhead is presumably much smaller for persistent
//   than non-persistent communication (or persistent would not add
//   value).
//  
//   data_time must dominate message_overhead for relevant cases or
//   persistent communication adds no value.
//  
//   Assuming persistent communication can add value,
//   messages should be large enough so that data_time dominates
//   for persistent communication but small enough so that
//   message_overhead dominates for non-persistent communication.
//   This determines a window of message sizes appropriate
//   for persistent communication.
//
template <class type>
class BlockCommunicator
{
 enum CommState
 {
   NONE=0,
   INITIAL,
   ACTIVE,
   FINISHED
 };
 private: // data
  std::list<void*>::iterator curr_block;
  // list of pointers avoids need for placement new
  std::list<void*> blockList;
  //std::list<Block<type>*>::iterator curr_block;
  //std::list<Block<type>*> blockList;
  Connection connection;
  int blocksize;
  // for generating the ID of the block as a list element
  int nextListID;
  // for generating the ID of the block as a message
  int nextCommID;
  int commState;
 // methods
 public: // construction
  BlockCommunicator():
    connection(),
    blocksize(0),
    nextListID(0),
    nextCommID(0),
    commState(NONE)
  {}
  BlockCommunicator(Connection connection_,
    int blocksize_, int numblocks)
  {
    commState=NONE;
    init(connection_,blocksize_,numblocks);
  }
  BlockCommunicator(Connection connection_)
  {
    commState=NONE;
    init(connection_);
  }
  void init(Connection connection_, int blocksize_, int numblocks);
  void init(Connection connection_)
  {
    init(connection_, Parameters::get_blockSize(), Parameters::get_numBlocks());
  }
 //protected: // if destructor is public then it should be virtual
  ~BlockCommunicator()
  {
    std::list<void*>::iterator i;
    for(i=blockList.begin(); i != blockList.end(); ++i)
      delete &fetch_block(i);
  }
 // information access
 //
 public:
  const Connection& get_connection()const{return connection;}
 private: // access
  // mimic behavior of a ring
  void increment_block(std::list<void*>::iterator& block_iter)
  {
    block_iter++;
    if(block_iter == blockList.end())
      block_iter = blockList.begin();
  }
  Block<type>& fetch_block(std::list<void*>::iterator block)
  {
    return *((Block<type>*)*block);
  }
  Block<type>& fetch_next_block(std::list<void*>::iterator block_iter)
  {
    increment_block(block_iter);
    return fetch_block(block_iter);
  }
 public: // access
  Block<type>& fetch_curr_block()
  {
    assert(curr_block != blockList.end());
    return fetch_block(curr_block);
  }
  bool comm_finished() const
  {
    return commState==FINISHED;
  }
  bool comm_active()
  {
    return get_curr_request() != MPI_REQUEST_NULL;
  }

 // operations
 //
 private:
  void increment_curr_block()
  {
    increment_block(curr_block);
  }
  //void rewind()
  //{
  //  curr_block = blockList.begin();
  //}
 public: // receiving operations

  // insert blocks prior to the current block
  // and call recv_block on them.
  //
  // If there is nowhere to write an incoming message, then I
  // suppose that a good MPI implementation would buffer it
  // somewhere until we post a receive that can accept it, rather
  // than blocking on completing the send.  If MPI provides a
  // mechanism to query how much such intermediate buffering is
  // happening then we could use it as a trigger to insert more
  // receive blocks, but I don't know of such a mechanism.
  //
  // Alternatively, I think that we can take the number of send
  // blocks that we need to have in order to keep from having
  // to wait for sends to complete as a reasonable indicator
  // of the number of receive blocks needed to avoid having to
  // buffer.  So when the sender adds more blocks, it can signal
  // the receiver to do likewise.
  //
  void insert_more_recv_blocks(int numblocks=1)
  {
    for(int i=0;i<numblocks;i++)
    {
      Block<type>* newBlock = new Block<type>(blocksize, nextListID++);
      blockList.insert(curr_block, newBlock);
      newBlock->recv_block(connection);
    }
  }
 public: // receiving operations
  //
  // if this returns true, then the user must follow up by
  // calling fetch_received_block(status);
  //
  bool test_recv_curr_block(MPI_Status& status)
  {
    return fetch_curr_block().test_comm(status);
  }
  // After the user has processed the block returned by this
  // call, he should call release_received_block() to
  // advance to the next block and free this block to
  // be used for receiving another message.
  //
  Block<type>& fetch_received_block(MPI_Status& status)
  {
    fetch_curr_block().shrink_received_block(status);
    fetch_curr_block().set_request(MPI_REQUEST_NULL);
    if(signal_hack())
    {
      if(fetch_curr_block().insert_flag_is_set())
      {
        // insert another block to receive particles
        // prior to the block that we just received
        dprintf("****** as requested, I am inserting a block ******");
        insert_more_recv_blocks();
      }
    }
    if(fetch_curr_block().finished_flag_is_set())
    {
      //dprintf("fetched final block from %d.%s of size %d",
      //   connection.rank(),
      //   connection.tag_name(),
      //   fetch_curr_block().size());
      commState=FINISHED;
    }
    return fetch_curr_block();
  }
  // reactivate recv request and advance to next block
  void release_received_block()
  {
    // post receive again and proceed to next block
    fetch_curr_block().recv_block(connection);
    increment_curr_block();
    assert(fetch_curr_block().is_active());
  }

  // cancel and free any open receive requests
  void cancel_recvs()
  {
    std::list<void*>::iterator b;
    for(b=blockList.begin(); b!=blockList.end();b++)
    {
      MPI_Request& pending_request = fetch_curr_block().fetch_request();
      if(pending_request!=MPI_REQUEST_NULL)
      {
        MPI_Cancel(&pending_request);
        MPI_Request_free(&pending_request);
      }
    }
  }

  MPI_Request get_curr_request()
  {
    if(comm_finished())
      return MPI_REQUEST_NULL;
    return fetch_curr_block().fetch_request();
  }
 public:
  void post_recvs()
  {
    std::list<void*>::iterator b;
    for(b=blockList.begin(); b != blockList.end(); ++b)
    {
      assert(fetch_block(b).is_inactive());
      fetch_block(b).recv_block(connection);
    }
    // reset curr_block to initial block
    curr_block = blockList.begin();
  }

 public:
  void recv_start()
  {
    // make sure that receives are posted on all block
    std::list<void*>::iterator b;
    for(b=blockList.begin(); b != blockList.end(); ++b)
    {
      assert(fetch_block(b).is_active());
    }

    // reset initial state
    commState=INITIAL;
  }

 // sending routines
 //
 private:
  // assumes that there is a block to send
  void send_curr_block()
  {
    // send the block
    fetch_curr_block().send_block(connection,nextCommID++);
    // proceed to next block
    increment_curr_block();
    // start communication
    send_start();
  }
 public: // interface
  // send a single particle.
  //
  // returns true iff block was sent
  // (whereupon the caller might want to check for incoming messages).
  bool send(const type& in)
  {
    //assert_le(fetch_curr_block.size(), fetch_curr_block().get_capacity());
    // append the particle to the block.
    fetch_curr_block().fast_push_back(in);
    //dprintf("sending particle %d in direction %d.%s",
    //  int(in.get_t()),connection.rank(), connection.tag_name());

    // if the block is full, send it
    if(__builtin_expect(fetch_curr_block().isfull(),false))
    {
      send_curr_block();
      return true;
    }
    return false;
  }
  // make sure that the current block does not have
  // a still-active send request on it
  void send_start()
  {
    // if current block is still communicating,
    // either wait for it or insert another block 
    // and use it instead.
    //
    if(!fetch_curr_block().test_comm())
    {
      if(signal_hack())
      {
        // if next block is still sending, insert another block to
        // use instead as next_block
        //
        Block<type>* newBlock = new Block<type>(blocksize, nextListID++);
        blockList.insert(curr_block,newBlock);
        curr_block--;
        assert(*curr_block == newBlock);
        dprintf("inserted new block %d",nextListID-1);

        // hack: set flag in currblock to tell receiver
        // to insert another block for receiving
        fetch_curr_block().set_insert_flag();
      }
      else
      {
        // wait for communication to finish sending
        // so that we can reuse this block
        //
        dprintf("waiting for send of block %d to complete.",
          fetch_curr_block().get_id());
        fetch_curr_block().waitfor_send();
        dprintf("done waiting for send to complete.");
      }
    }
    fetch_curr_block().clear();
    assert(!fetch_curr_block().is_active());
    assert(fetch_curr_block().test_comm());
    commState=INITIAL;
  }
  // send the remaining particles, sending an empty message if
  // necessary to signal that no more particles will be sent.
  //
  // this assumes that curr_block exists and contains all unsent particles
  void send_complete()
  {
    assert(!fetch_curr_block().isfull());
    fetch_curr_block().set_finished_flag();
    send_curr_block();
  }
};

// === code above this point could go in something like BlockCommunicatorFwd.h ===
#include "Parameters.h"

template <typename type>
void BlockCommunicator<type>::init(Connection connection_, int blocksize_, int numblocks)
{
  assert(commState==NONE);
  connection = connection_;
  blocksize = blocksize_;
  commState = INITIAL;
  nextCommID = 0;

  assert(blocksize>0);
  assert(numblocks>0);
  for(nextListID=0;nextListID<numblocks;nextListID++)
  {
    Block<type>* newBlock = new Block<type>(blocksize, nextListID);
    blockList.push_back(newBlock);
  }
  curr_block = blockList.begin();
}

//#include <iosfwd> // for ostream
//
//inline std::ostream& operator<<(std::ostream& os, const Block<Particle>& block_)
//{
//  const aligned_vector(Particle)& block = block_.get_block();
//  for(int k=0; k<block.size();k++)
//  {
//    os << "\n  block[" << k << "] = " << block[k];
//  }
//  return os;
//}

#endif
