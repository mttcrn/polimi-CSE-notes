# Threat Modeling
Security process as solving the objective of "obtaining a goal against an adversary".
It is achievable trough:
- Policies, the goals/objectives that the system should enforce.
- Threat Model, a set of assumptions on the adversary.
- Mechanism, all the measures that ensures the policies are followed as long as the threat model is correct. 

Since security is a negative goal, we need TMs: we should ensure that the policies are followed independently from the external world.  
It is much harder to validate than a positive goal, as it requires to think about every possibile adversary and their attacks. For this reason, TMs end up being a set of assumption on the attacker's capabilities.

Mechanism are security measures. They ensure that the policies are followed as long as the assumptions on the attackers are correct. 
Mechanism includes more than mere technological solutions: they integrate all the possible ways in which attacks may be handled. 
Often, mechanism are simple. It's mostly a matter of recognizing the necessity to use them. 

Threat modeling helps in identifying the three components:
$$
Risk = Assets \times Vulnerabilities \times Threats
$$
We will still have to handle the problems of costs, feasibility, user experience, safety, and so on.

![](./assets/how_to_TM.png)

## Model the system
Represents the (data) flows and interactions amongst the entities of the system. We can insert boundaries to identify who controls each part of the system, and identify which are trust boundaries. 

The system may be huge, but not representing it all may not show some threats. The knowledge of the system may not be centralized in real world scenarios. At different level of abstraction we may identify different details. 

### Data Flow Diagrams

![[data_flow_diagram.png]]

| Element          | Appereance                                | Meaning                                                                  | Examples                                                         |
| ---------------- | ----------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| External entity  | Rectangle with sharp corners              | People, code, and other stuff outside your control                       | Employees, customers, other organization with which you interact |
| Process          | Rounded rectangle / Concentric rectangles | Any running agent, usually code                                          | Browser, HTTP server, ADAS perception software                   |
| Data flow        | Arrow, mono- or bi- directional           | Communication and interacation between other components                  | Network connections, HTTP, Modbus                                |
| Data store       | Two parallel lines                        | Stuff to store data                                                      | Files, Databases, Memory                                         |
| Trust boundaries | Dotted lines or dotted rectangle          | Represent where your "control" ends, where the interactions are not safe |                                                                  |
### Swim Lane Diagrams
![](./assets/swim_lane_diagram.png)
Lanes (columns) are actors or components. Arrow represent the flow of information between lines. 
It is useful to represent time-dependent interactions (e.g. networks protocol).
It may be useful to identify trust boundaries, if some of the interactions are between trusted sources. 

From a practical point of view, a model needs to be comprehensible. Multiple sub-models may be added for each "relevant" component.
## Find, Address and Validate threats
Threats themselves can be modeled by brainstorming, using structured approaches or using attack trees. 
### Brainstorming
It is the easiest way to threat model. We get a bunch of experts in a room and make them
hypothesize attacks and threats against the system. 
However it may leads to unbalanced threats depending on the expertise, or impossible-to-realize complex attacks, ignoring simpler but less “interesting” threats.
### Structured Approaches
#### STRIDE
The acronym stands for: 
- **S**poofing: it means imitating, pretending to be something you are not. In (cyber)security, it describes an attack where a malicious actor attempts to disguise a data flow as coming from a legitimate (trusted) source, when in fact it does not. 
  It violates **authentication**.
	- Weak authentication schemes, tampering of certificates, controlling the victim's machine, may all lead the attacker to present a file as authenticated while it's not.
	- [Wifi evil attack](https://www.kaspersky.com/resource-center/preemptive-safety/evil-twin-attacks).
	  Wifi are identified with their SSID. In WPA2 the AP authenticates the device: the AP with stronger signal is preferred. If the attacker knows the password (e.g. public places), he/she can create an evil twin of the SSID with the correct password to make the victim's device automatically connect, thus becoming a MITM.
- **T**ampering: it means modifying, and in (cyber)security it implies the (unallowed/unexpected) modification of data either at rest or in transit. 
  It threats **integrity** of data.
  It is easy to confuse with spoofing, and sometimes it overlaps with it. 
	- The easiest examples is buffer overflow in which an attacker modify a region of memory that he/she should not be allowed to access. 
	- Another example is the well known bank withdrawal race condition, since the saved result is incorrect due to an error in the sequence of executed instruction.
	- Hardware attacks such as Row hammer modify directly bits in memory. There is no authentication of the user. However, some sort of "access" to the memory of the device is necessary. 
	- Time-Of-Check vs Time-Of-Use (TOCTOU) bugs.
	  If setuid is set, the attacker can change the “file” pointer with a symlink between check and usage. The wrong assumption is that the the state managed by the operating system (in this case the file system namespace) will not change between system calls.
	  Similar problem may arise in network contexts. 
	- Network tampering is not limited to modification of already sent packets, but can comprehend their addition or removal. Big overlap with "spoofing machines".
	- Encryption schemes can be applied either at the channel level (e.g. VPNs) or at message level (e.g. signatures).
	- In communication schemes, the message does not disappear once the channel has finished, so encrypting only the channel would leave part of the system undefended. If the channels are multiple and interchangeable, we would ensure that at least one is encrypted at every hop. Moreover, the message encryption requires compatibility on both end applications. 
	- Tampering with the perceived time of the machine may lead to extend validation or the opposite, if the goal is to deny functionalities (e.g. sessions, certificates, licenses expire).
	- AI/ML: modifying the training set allows to lower the precision, inject backdoors, or targeted misscalssification of inputs. 
- **R**epudiation: in (cyber)security it is a threat to **accountability** where an individual, entity, or system denies having performed an action, leaving the system unable to prove otherwise. 
  It is strictly related with the concept of fraud although not exactly overlapping. It often works against privacy. 
	- Messages may have never been sent, received or read. It a critical interaction necessitates to make a party liable for its actions, what happens if such actions are not provable?
	  Multiple public persons claimed their social accounts were hacked after personal messages or public tweets raised scandals.
	  The reason for which [swatting](https://en.wikipedia.org/wiki/Swatting) became popular is that it is actually hard to prove the ownership of the call. Therefore, it is a low risk attack.
	- Fraud through repudiation can be executed by the seller, the buyer, or any intermediate. 
	  What if someone claims that the ATM didn’t give them their money? 
	  The attacker may be the bank (claiming they did but never providing you the money) the intermediary (e.g., an ATM provider not from the bank, keeping the money for themselves) or the user, receiving the money and claiming they were not provided (or not even attempting the transaction).
	- Something similar happens in e-commerce.
	  The user would like to be sure that if they receive something incorrect or never receive anything, they’ll be provided either their money or the goods. The seller (not the service provider) would like to be sure that the money will be received. 	  Each one of the intermediaries wants to be sure that they are not given  responsibility of the “lost” or “modified” package.
	  Some e-markets accept it as an economic risk, force the seller to take at least part of the responsibility while the user is more protected being the client. Local laws also affect such decisions, some american states allow to drop packages at the door with a photograph. Overall, managing fraud entails logging various events.
	- Logging software is yet another piece of code, hence it expands the attack surface of the system. 
	  TOR, VPNs and browser privacy extensions tamper with information about the actual sender. In general, logs may be directly or indirectly tampered with by an attacker. Think about rootkits, or adversarial attacks against intrusion detection systems. 
	- Deepfakes and in general video, photo and audio digital editing allow for a recorded person to claim that the data is altered. If video and audios can be faked, how do we prove recorded actions?
- **I**nformation disclosure: in (cyber)security it is the process of allowing entities to see information that they are not authorized to see. 
  It is a threat against **confidentiality**.
  Data can be seen either at rest or in motion (it could also be seen as data in use).
	- Physical disk data recovery (freed memory is not intrinsically overwritten).
	- Metadata may be as relevant as data itself. It is not only the content of the file that may disclose information, but directly the file name or folder themselves. 
	- Data may need to be decrypted to be used. The basic structure of such attack is to dump all RAM to retrieve data stored by others, but of course others’ memory should not be accessed by an unprivileged user. However, many techniques have been studied for memory dumping.
	- What is sensible information is not trivial (e.g. the IP of our packets is in clear, even when we use TLS). Communication per-se may be something that we don’t want to share (TOR / VPNs). The most obvious attack is sniffing.
	- Spectre vulnerability
	  Branch prediction allows CPUs to speed up computation, loading data in memory for the most commonly taken branch. The attacker “teaches” the branch predictor to expect a specific pattern, and then forces an invalid action which is denied, but the branch predictor still loads it in memory. A memory dump allows to retrieve the information.
	- Timing attacks exploit variations in the execution time of cryptographic or security-sensitive operations to infer secret information, such as cryptographic keys or passwords.
- **D**enial of service: in (cyber)security it is the threat that aims at absorbing/consuming one or more resources a to deny a service from working/reaching the user. It is always physical, with infinite resources it will not happen. 
  It is a threat against **availability**.
	- Computation exhaustion
	  At the core of information system there are running process: it is sufficient to stop a process to deny the service. 
	  DoS can be implemented trough "crashing" the target or exhausting it. Crashing it is heavily dependent on the specific implementation vulnerabilities. Exhausting computation may be done through requesting unbalanced tasks to the server (e.g., video processing, encryption/decryption routines, ..).
	- Zip bombs may also aim at exhaust computation, but most commonly just exhaust disk storage.
	- Network exhaustion is performed either through amplification factors or “crashes”, networks are insecure. Some examples are smurfs, ping of death, land attacks, amplification hell.
	- Fork bombs work similarly to Zip bombs, a program forks infinitely to exhaust PIDs.
	- Money can be the exhausted resource (e.g., SMSs, cloud computation).
	- Electricity is particularly relevant in IoT environments, or in space.
	- Note that countermeasures often ends up having the downside of enabling DoS: locking a device after a number of login attempts, firewalls automatically banning IPs on pattern detection, rate limiting on APIs or services, reputation-based email filtering, and so on. 
- **E**levation of privilege (or Expansion of Authority): it means allowing someone to do something that they are not authorized to do. Many examples that we already provided may have an Elevation of Privilege effect.
	- SQL injections, buffer overflows, and Log4Shell make the attacker capable of executing commands with privileges they should not own.
	- An attacker obtaining access to a local network may interact with services they should not have the authority to interact with.
	- A confused deputy is an entity with certain privileges that is tricked into misusing its authority on behalf of another (less or differently privileged) user, leading it to perform operations that the user is not authorized to execute. 
	  It is a perfect case of social engineering.
	  Cross Site Request Forgery attacks force the browser to act as a confused deputy, sending the attacker’s commands with the authorization of the user.
	  Shellcode execution in SUID processes allows the attacker to execute code as the owner of the file. 
	  Forcing the printing of relevant information in debug or error logs, similarly, convinces the “process” to provide the attacker with information that they should not have.

STRIDE-per-elements consider the case that only some elements can have some threat categories. The core limitation derives from having to do the data flow diagram (DFD).
STRIDE-per-interactions assumes all threats can be represented in the interaction between elements, since the element by itself does not "act". Limited if an element is not fully expanded, since the interaction between the internal systems of the element would not be visible. 

|                 | S   | T   | R      | I   | D   | E   |
| --------------- | --- | --- | ------ | --- | --- | --- |
| External Entity | yes |     | yes    |     |     |     |
| Process         | yes | yes | yes    | yes | yes | yes |
| Data Store      |     | yes | yes/no | yes | yes |     |
| Data            |     | yes |        | yes | yes |     |
### MITRE's ATT&CK
MITRE ATT&CK (Adversarial Tactics, Techniques, and Common Knowledge) is a publicly available framework that compiles observed adversary behaviours.
The framework is organized as a matrix:
- Columns represent tactics (why) - the adversary's objective at a given phase of the attack. 
- Rows represents techniques (how) - methods that an adversary may use to accomplish the foal. 
For each technique the framework offers: examples of how real-world attackers used it, mitigation strategies to monitor or prevent their impact. 

|      | STRIDE                                    | MITRE ATT&CK<br>& other knowledge bases                                |
| ---- | ----------------------------------------- | ---------------------------------------------------------------------- |
| Pros | Simple<br>Comprehensive                   | Based on "real" data<br>Specific<br>Easy to understand countermeasures |
| Cons | High-level<br>Not specific<br>Theoretical | Complex<br>Does not consider "0 days"<br>Not applicable to everything  |
### Attack Trees
Kind of deprecated.
The core idea is to develop a tree of sequential events to reach a goal. Then, prune the tree depending on the system's characteristics. 
The problem is that too many attack trees for a single system are needed, plus an excessive complexity make them hard to use. 
Still they present an interesting overview of all the necessary/possible steps to achieve an attack goal.
![](./assets/attack_tree.png)

### Mitigating Threats
- **Top-Down Approach**: start with a system-wide view, then analyze components. This ensures all potential threats are considered.
- **Breadth-First**: evaluate all threats at a given level before diving into lower layers. Unmitigated threats at higher layers are often more critical.
- **Threat-Mitigation Hierarchy**: every mitigation introduces new attack vectors, requiring additional defenses.

- **Avoid the risk**: if the risk is greater than the potential reward, you can decide not to implement the software, application or feature.
- **Address the risk**: after identifying the threat, you change the design or operational processes, or implement mitigations, to reduce the risk. 
- **Accept the risk**: if the threat is real but has low probability or impact, you can decide to accept the costs of something going wrong.
- **Transfer the risk**: If the threat is something that someone else is better equipped to handle, you can make them responsible for it, like providing end-users or customers a terms of service and licensing agreement or purchasing insurance.
- **Ignore the risk**: If you want to pretend the threat doesn’t exist, then you can ignore it, but this decision often means you open yourself up to legal liability or compliance violations.

The attacker has room in the gap between what a system is designed to do, and what it does to achieve its goal. Our perfect goal would be to design a system that does what it is designed for. 